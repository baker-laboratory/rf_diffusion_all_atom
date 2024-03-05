"""
Inference script.

To run with aa.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name other_config

where other_config can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import os

import re
import os, time, pickle
import dataclasses
import torch 
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from util import writepdb_multi, writepdb
from inference import utils as iu
from icecream import ic
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob
import inference.model_runners
import rf2aa.tensor_util
import idealize_backbone
import rf2aa.util
import aa_model
import copy

import e3nn.o3 as o3

def warm_up_spherical_harmonics():
    ''' o3.spherical_harmonics returns different values on 1st call vs all subsequent calls
    All subsequent calls are reproducible.
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    relative_pos = torch.tensor([[1.,1.,1.], [1.,1.,1.]]).to(device).to(torch.float32)
    sh1 = o3.spherical_harmonics([1,2,3], relative_pos, normalize=True)
    sh2 = o3.spherical_harmonics([1,2,3], relative_pos, normalize=True)

def make_deterministic(seed=0):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    seed_all(seed)
    warm_up_spherical_harmonics()

def seed_all(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_seeds():
    return {
        'torch': torch.get_rng_state(),
        'np': np.random.get_state(),
        'python': random.getstate(),
    }

@hydra.main(version_base=None, config_path='config/inference', config_name='aa')
def main(conf: HydraConfig) -> None:
    sampler = get_sampler(conf)
    sample(sampler)

def get_sampler(conf):
    if conf.inference.deterministic:
        make_deterministic()

    # Loop over number of designs to sample.
    design_startnum = conf.inference.design_startnum
    if conf.inference.design_startnum == -1:
        existing = glob.glob(conf.inference.output_prefix + '*.trb')
        indices = [-1]
        for e in existing:
            m = re.match(r'.*_(\d+)\.trb$', e)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1   

    conf.inference.design_startnum = design_startnum
    # Initialize sampler and target/contig.
    sampler = inference.model_runners.sampler_selector(conf)
    return sampler


def sample(sampler):

    log = logging.getLogger(__name__)
    des_i_start = sampler._conf.inference.design_startnum
    des_i_end = sampler._conf.inference.design_startnum + sampler.inf_conf.num_designs
    for i_des in range(sampler._conf.inference.design_startnum, sampler._conf.inference.design_startnum + sampler.inf_conf.num_designs):
        if sampler._conf.inference.deterministic:
            seed_all(i_des)

        start_time = time.time()
        out_prefix = f'{sampler.inf_conf.output_prefix}_{i_des}'
        sampler.output_prefix = out_prefix
        log.info(f'Making design {out_prefix}')
        if sampler.inf_conf.cautious and os.path.exists(out_prefix+'.pdb'):
            log.info(f'(cautious mode) Skipping this design because {out_prefix}.pdb already exists.')
            continue
        log.info(f'making design {i_des} of {des_i_start}:{des_i_end}')
        sampler_out = sample_one(sampler)
        log.info(f'Finished design in {(time.time()-start_time)/60:.2f} minutes')
        save_outputs(sampler, out_prefix, *sampler_out)

def sample_one(sampler, simple_logging=False):
    # For intermediate output logging
    indep = sampler.sample_init()

    denoised_xyz_stack = []
    px0_xyz_stack = []
    seq_stack = []

    rfo = None

    # Loop over number of reverse diffusion time steps.
    for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step-1, -1):
        if simple_logging:
            e = '.'
            if t%10 == 0:
                e = t
            print(f'{e}', end='')
        px0, x_t, seq_t, tors_t, plddt, rfo = sampler.sample_step(
            t, indep, rfo)
        rf2aa.tensor_util.assert_same_shape(indep.xyz, x_t)
        indep.xyz = x_t
            
        aa_model.assert_has_coords(indep.xyz, indep)

        px0_xyz_stack.append(px0)
        denoised_xyz_stack.append(x_t)
        seq_stack.append(seq_t)
    
    # Flip order for better visualization in pymol
    denoised_xyz_stack = torch.stack(denoised_xyz_stack)
    denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
    px0_xyz_stack = torch.stack(px0_xyz_stack)
    px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])

    return indep, denoised_xyz_stack, px0_xyz_stack, seq_stack


def save_outputs(sampler, out_prefix, indep, denoised_xyz_stack, px0_xyz_stack, seq_stack):
    log = logging.getLogger(__name__)

    final_seq = seq_stack[-1]

    if sampler._conf.seq_diffuser.seqdiff is not None:
        # When doing sequence diffusion the model does not make predictions beyond category 19
        final_seq = final_seq[:,:20] # [L,20]

    # All samplers now use a one-hot seq so they all need this step
    final_seq[~indep.is_sm, 22:] = 0
    final_seq = torch.argmax(final_seq, dim=-1)

    # replace mask and unknown tokens in the final seq with alanine
    final_seq = torch.where((final_seq == 20) | (final_seq==21), 0, final_seq)
    seq_design = final_seq.clone()
    xyz_design = denoised_xyz_stack[0].clone()

    # Determine lengths of protein and ligand for correct chain labeling in output pdb
    chain_Ls = rf2aa.util.Ls_from_same_chain_2d(indep.same_chain)

    # Save outputs
    out_head, out_tail = os.path.split(out_prefix)
    unidealized_dir = os.path.join(out_head, 'unidealized')
    os.makedirs(out_head, exist_ok=True)
    os.makedirs(unidealized_dir, exist_ok=True)

    # pX0 last step
    out_unidealized = os.path.join(unidealized_dir, f'{out_tail}.pdb')
    xyz_design[sampler.is_diffused, 3:] = np.nan
    aa_model.write_traj(out_unidealized, xyz_design[None,...], seq_design, indep.bond_feats, chain_Ls=chain_Ls, lig_name=sampler._conf.inference.ligand, idx_pdb=indep.idx)
    out_idealized = f'{out_prefix}.pdb'

    aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, out_unidealized)

    # Idealize the backbone (i.e. write the oxygen at the position inferred from N,C,Ca)
    idealize_backbone.rewrite(out_unidealized, out_idealized)
    des_path = os.path.abspath(out_idealized)

    # trajectory pdbs
    traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)
    os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

    out = f'{traj_prefix}_Xt-1_traj.pdb'
    aa_model.write_traj(out, denoised_xyz_stack, final_seq, indep.bond_feats, chain_Ls=chain_Ls, lig_name=sampler._conf.inference.ligand, idx_pdb=indep.idx)
    xt_traj_path = os.path.abspath(out)

    out=f'{traj_prefix}_pX0_traj.pdb'
    aa_model.write_traj(out, px0_xyz_stack, final_seq, indep.bond_feats, chain_Ls=chain_Ls, lig_name=sampler._conf.inference.ligand, idx_pdb=indep.idx)
    x0_traj_path = os.path.abspath(out)

    # run metadata
    sampler._conf.inference.input_pdb = os.path.abspath(sampler._conf.inference.input_pdb)
    trb = dict(
        config = OmegaConf.to_container(sampler._conf, resolve=True),
        device = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU',
        px0_xyz_stack = px0_xyz_stack.detach().cpu().numpy(),
        indep={k:v.detach().cpu().numpy() for k,v in dataclasses.asdict(indep).items()},
    )
    if hasattr(sampler, 'contig_map'):
        for key, value in sampler.contig_map.get_mappings().items():
            trb[key] = value

    for out_path in des_path, xt_traj_path, x0_traj_path:
        aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, out_path)

    with open(f'{out_prefix}.trb','wb') as f_out:
        pickle.dump(trb, f_out)

    log.info(f'design : {des_path}')
    log.info(f'Xt traj: {xt_traj_path}')
    log.info(f'X0 traj: {x0_traj_path}')


if __name__ == '__main__':
    main()
