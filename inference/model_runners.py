from contextlib import contextmanager
from datetime import datetime
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from rf2aa.chemical import ChemicalData as ChemData
from icecream import ic

import rf2aa.chemical
import rf2aa.util
from rf2aa.util_module import XYZConverter
import rf2aa.tensor_util
import aa_model

from diffusion import Diffuser
from contigs import ContigMap
from inference import utils as iu
from potentials.manager import PotentialManager
import logging
import util
from hydra.core.hydra_config import HydraConfig
from rf2aa.model import RoseTTAFoldModel

import sys
sys.path.append('../') # to access RF structure prediction stuff 

from rf2aa.model import RoseTTAFoldModel

TOR_INDICES  = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES   = util.reference_angles

class Sampler:

    def __init__(self, conf: DictConfig):
        """Initialize sampler.
        Args:
            conf: Configuration.
        """
        self.initialized = False
        self.initialize(conf)
    
    def initialize(self, conf: DictConfig):
        self._log = logging.getLogger(__name__)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        needs_model_reload = not self.initialized or conf.inference.ckpt_path != self._conf.inference.ckpt_path

        # Assign config to Sampler
        self._conf = conf

        # Initialize inference only helper objects to Sampler
        self.ckpt_path = conf.inference.ckpt_path

        if needs_model_reload:
            # Load checkpoint, so that we can assemble the config
            self.load_checkpoint()
            self.assemble_config_from_chk()
            # Now actually load the model weights into RF
            self.model = self.load_model()
        else:
            self.assemble_config_from_chk()

        self.initialized=True

        # Initialize helper objects
        self.inf_conf = self._conf.inference
        self.contig_conf = self._conf.contigmap
        self.denoiser_conf = self._conf.denoiser
        self.ppi_conf = self._conf.ppi
        self.potential_conf = self._conf.potentials
        self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess
        self.diffuser = Diffuser(**self._conf.diffuser)
        self.model_adaptor = aa_model.Model(self._conf)
        # Temporary hack
        self.model.assert_single_sequence_input = True
        self.model_adaptor.model = self.model

        self.converter = XYZConverter()
        self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=False, center=False)
        self.chain_idx = None

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        if self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs:
            self.ppi_conf.binderlen = ''.join(chain_idx[0] for chain_idx in self.target_feats['pdb_idx']).index('B')

        self.potential_manager = PotentialManager(self.potential_conf, 
                                                  self.ppi_conf, 
                                                  self.diffuser_conf, 
                                                  self.inf_conf)
        
        # Get recycle schedule    
        recycle_schedule = str(self.inf_conf.recycle_schedule) if self.inf_conf.recycle_schedule is not None else None
        self.recycle_schedule = iu.recycle_schedule(self.T, recycle_schedule, self.inf_conf.num_recycles)

    def process_target(self, pdb_path):
        assert not (self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs), "target reprocessing not implemented yet for these configuration arguments"
        self.target_feats = iu.process_target(self.inf_conf.input_pdb)
        self.chain_idx = None

    @property
    def T(self):
        '''
            Return the maximum number of timesteps
            that this design protocol will perform.

            Output:
                T (int): The maximum number of timesteps to perform
        '''
        return self.diffuser_conf.T
    
    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        self._log.info(f'Reading checkpoint from {self.ckpt_path}')
        print(f'loading {self.ckpt_path}')
        self.ckpt  = torch.load(
            self.ckpt_path, map_location=self.device)
        print(f'loaded {self.ckpt_path}')

    def assemble_config_from_chk(self) -> None:
        
        # get overrides to re-apply after building the config from the checkpoint
        overrides = []
        if HydraConfig.initialized():
            overrides = HydraConfig.get().overrides.task
        if 'config_dict' in self.ckpt.keys():
            # First, check all flags in the checkpoint config dict are in the config file
            for cat in ['model','diffuser','seq_diffuser','preprocess']:
                for key in self._conf[cat]:
                    if key == 'chi_type' and self.ckpt['config_dict'][cat][key] == 'circular':
                        continue
                    try:
                        self._conf[cat][key] = self.ckpt['config_dict'][cat][key]
                    except:
                        pass
            # add back in overrides again
            for override in overrides:
                if override.split(".")[0] in ['model','diffuser','seq_diffuser','preprocess']:
                    print(f'OVERRIDING: You are changing {override.split("=")[0]} from the value this model was trained with.')
                    mytype = type(self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])
                    self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(override.split("=")[1])
        else:
            print('WARNING: Model, Diffuser and Preprocess parameters are not saved in this checkpoint. Check carefully that the values specified in the config are correct for this checkpoint')       

    def load_model(self):
        """Create RosettaFold model from preloaded checkpoint."""

        # for all-atom str loss
        self.ti_dev = ChemData().torsion_indices
        self.ti_flip = ChemData().torsion_can_flip
        self.ang_ref = ChemData().reference_angles
        self.fi_dev = ChemData().frame_indices
        self.l2a = ChemData().long2alt
        self.aamask = ChemData().allatom_mask
        self.num_bonds = ChemData().num_bonds
        self.atom_type_index = ChemData().atom_type_index
        self.ljlk_parameters = ChemData().ljlk_parameters
        self.lj_correction_parameters = ChemData().lj_correction_parameters
        self.hbtypes = ChemData().hbtypes
        self.hbbaseatoms = ChemData().hbbaseatoms
        self.hbpolys = ChemData().hbpolys
        self.cb_len = ChemData().cb_length_t
        self.cb_ang = ChemData().cb_angle_t
        self.cb_tor = ChemData().cb_torsion_t

        # model_param.
        self.ti_dev = self.ti_dev.to(self.device)
        self.ti_flip = self.ti_flip.to(self.device)
        self.ang_ref = self.ang_ref.to(self.device)
        self.fi_dev = self.fi_dev.to(self.device)
        self.l2a = self.l2a.to(self.device)
        self.aamask = self.aamask.to(self.device)
        self.num_bonds = self.num_bonds.to(self.device)
        self.atom_type_index = self.atom_type_index.to(self.device)
        self.ljlk_parameters = self.ljlk_parameters.to(self.device)
        self.lj_correction_parameters = self.lj_correction_parameters.to(self.device)
        self.hbtypes = self.hbtypes.to(self.device)
        self.hbbaseatoms = self.hbbaseatoms.to(self.device)
        self.hbpolys = self.hbpolys.to(self.device)
        self.cb_len = self.cb_len.to(self.device)
        self.cb_ang = self.cb_ang.to(self.device)
        self.cb_tor = self.cb_tor.to(self.device)

        model = RoseTTAFoldModel.RoseTTAFoldModule(
            **self._conf.model,
            aamask=self.aamask,
            atom_type_index=self.atom_type_index,
            ljlk_parameters=self.ljlk_parameters,
            lj_correction_parameters=self.lj_correction_parameters,
            num_bonds=self.num_bonds,
            cb_len = self.cb_len,
            cb_ang = self.cb_ang,
            cb_tor = self.cb_tor,
            assert_single_sequence_input=True,
            ).to(self.device)
        
        model = model.eval()
        self._log.info(f'Loading checkpoint.')
        if not self._conf.inference.zero_weights:
            model.load_state_dict(self.ckpt[self._conf.inference.state_dict_to_load], strict=True)
        return model

    def construct_contig(self, target_feats):
        """Create contig from target features."""
        if self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs:
            seq_len = target_feats['seq'].shape[0]
            self.contig_conf.contigs = [f'{self.ppi_conf.binderlen}',f'B{self.ppi_conf.binderlen+1}-{seq_len}']
        self._log.info(f'Using contig: {self.contig_conf.contigs}')
        # self.contig_conf.contigs = ['']
        if self.contig_conf.contigs == 'whole':
            L = len(target_feats["pdb_idx"])
            self.contig_conf.contigs = [f'{L}-{L}']
        return ContigMap(target_feats, **self.contig_conf)

    def construct_denoiser(self, L, visible):
        """Make length-specific denoiser."""
        denoise_kwargs = OmegaConf.to_container(self.diffuser_conf)
        denoise_kwargs.update(OmegaConf.to_container(self.denoiser_conf))
        aa_decode_steps = min(denoise_kwargs['aa_decode_steps'], denoise_kwargs['partial_T'] or 999)
        denoise_kwargs.update({
            'L': L,
            'diffuser': self.diffuser,
            'seq_diffuser': None,
            'potential_manager': self.potential_manager,
            'visible': visible,
            'aa_decode_steps': aa_decode_steps,
        })
        return iu.Denoise(**denoise_kwargs)

    def sample_init(self, return_forward_trajectory=False):
        """Creates initial features to start the sampling process."""

        # moved this here as should be updated each iteration of diffusion
        self.contig_map = self.construct_contig(self.target_feats)
        L = len(self.target_feats['pdb_idx'])

        indep_orig = aa_model.make_indep(self._conf.inference.input_pdb, self._conf.inference.ligand)
        indep, self.is_diffused, self.is_seq_masked = self.model_adaptor.insert_contig(indep_orig, self.contig_map)
        self.t_step_input = self._conf.diffuser.T
        if self.diffuser_conf.partial_T:
            mappings = self.contig_map.get_mappings()
            assert indep.xyz.shape[0] ==  L + torch.sum(indep.is_sm), f"there must be a coordinate in the input PDB for each residue implied by the contig string for partial diffusion.  length of input PDB != length of contig string: {indep.xyz.shape[0]} != {L+torch.sum(indep.is_sm)}"
            assert torch.all(self.is_diffused[indep.is_sm] == 0), f"all ligand atoms must be in the motif"
            assert (mappings['con_hal_idx0'] == mappings['con_ref_idx0']).all(), 'all positions in the input PDB must correspond to the same index in the output pdb'
            indep = indep_orig
        indep.seq[self.is_seq_masked] = ChemData().MASKINDEX
        # Diffuse the contig-mapped coordinates 
        if self.diffuser_conf.partial_T:
            self.t_step_input = self.diffuser_conf.partial_T
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
        t_list = np.arange(1, self.t_step_input+1)
        atom_mask = None
        seq_one_hot = None
        fa_stack, xyz_true = self.diffuser.diffuse_pose(
            indep.xyz,
            seq_one_hot,
            atom_mask,
            indep.is_sm,
            diffusion_mask=~self.is_diffused,
            t_list=t_list,
            diffuse_sidechains=self.preprocess_conf.sidechain_input,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input)

        xT = fa_stack[-1].squeeze()[:,:14,:]
        xt = torch.clone(xT)
        indep.xyz = xt

        self.denoiser = self.construct_denoiser(len(self.contig_map.ref), visible=~self.is_diffused)
        
        self.msa_prev = None
        self.pair_prev = None
        self.state_prev = None
        
        return indep

class NRBStyleSelfCond(Sampler):
    """
    Model Runner for self conditioning in the style attempted by NRB
    """

    def sample_step(self, t, indep, rfo):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.
        '''

        rfi = self.model_adaptor.prepro(indep, t, self.is_diffused)
        rf2aa.tensor_util.to_device(rfi, self.device)
        seq_init = torch.nn.functional.one_hot(
                indep.seq, num_classes=ChemData().NAATOKENS).to(self.device).float()
        seq_t = torch.clone(seq_init)

        # Self conditioning
        if ((t < self.diffuser.T) and (t != self.diffuser_conf.partial_T)) and self._conf.inference.str_self_cond:
            rfi = aa_model.self_cond(indep, rfi, rfo)

        with torch.no_grad():
            with reset_and_increment_numpy_rng(1):
                rfo = self.model_adaptor.forward(
                                    rfi,
                                    return_infer=True,
                                    )

        px0 = rfo.get_xyz()[:,:14]
        logits = rfo.get_seq_logits()

        # Default method of decoding sequence
        seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
        sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position

        pseq_0 = torch.nn.functional.one_hot(
            sampled_seq, num_classes=ChemData().NAATOKENS).to(self.device).float()
        
        pseq_0[~self.is_seq_masked] = seq_init[~self.is_seq_masked].to(self.device) # [L,22]

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        self._log.info(
                f'{current_time}: Timestep {t}, current sequence: { ChemData().seq2chars(torch.argmax(pseq_0, dim=-1).tolist())}')

        if t > self._conf.inference.final_step:
            x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
                xt=rfi.xyz[0,:,:14].cpu(),
                px0=px0,
                t=t,
                diffusion_mask=~self.is_diffused,
                seq_diffusion_mask=~self.is_diffused,
                seq_t=seq_t,
                pseq0=pseq_0,
                diffuse_sidechains=self.preprocess_conf.sidechain_input,
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
            )
        else:
            # Final step.
            px0 = px0.cpu()
            px0[~self.is_diffused] = indep.xyz[~self.is_diffused]
            x_t_1 = torch.clone(px0)
            seq_t_1 = pseq_0

            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.is_diffused.shape[-1], 10, 2))

        px0 = px0.cpu()
        x_t_1 = x_t_1.cpu()
        seq_t_1 = seq_t_1.cpu()

        return px0, x_t_1, seq_t_1, tors_t_1, None, rfo

def sampler_selector(conf: DictConfig):
    if conf.inference.model_runner == 'NRBStyleSelfCond':
        sampler = NRBStyleSelfCond(conf)
    else:
        raise ValueError(f'Unrecognized sampler {conf.model_runner}')
    return sampler

@contextmanager
def reset_and_increment_numpy_rng(n):
    state = np.random.get_state()
    yield
    np.random.set_state(state)
    for _ in range(n):
        np.random.rand(1)