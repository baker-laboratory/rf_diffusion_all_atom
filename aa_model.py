import torch
import assertpy
from collections import defaultdict
import torch.nn.functional as F
import dataclasses
from icecream import ic
from assertpy import assert_that
from omegaconf import OmegaConf
import rf2aa.util
from dataclasses import dataclass
from rf2aa.kinematics import get_chirals
from rf2aa.util_module import XYZConverter
import rf2aa.tensor_util
import torch
import copy
import numpy as np
from kinematics import get_init_xyz
import chemical
import rf2aa.chemical
from rf2aa.data.parsers import parse_mol, parse_pdb
import util
import inference.utils
import networkx as nx
import pdbio

from rf2aa.data.data_loader import get_bond_distances
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.chemical import initialize_chemdata
initialize_chemdata(OmegaConf.create({'use_phospate_frames_for_NA': True}))

NINDEL=1
NTERMINUS=2
NMSAFULL=ChemData().NAATOKENS+NINDEL+NTERMINUS
NMSAMASKED=ChemData().NAATOKENS+ChemData().NAATOKENS+NINDEL+NINDEL+NTERMINUS

MSAFULL_N_TERM = ChemData().NAATOKENS+NINDEL
MSAFULL_C_TERM = MSAFULL_N_TERM+1

MSAMASKED_N_TERM = 2*ChemData().NAATOKENS + 2*NINDEL
MSAMASKED_C_TERM = 2*ChemData().NAATOKENS + 2*NINDEL + 1

N_TERMINUS = 1
C_TERMINUS = 2

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
def chain_letters_from_same_chain(same_chain):
    L = same_chain.shape[0]
    G = nx.from_numpy_array(same_chain.numpy())
    cc = list(nx.connected_components(G))
    cc.sort(key=min)
    chain_letters = np.chararray((L,), unicode=True)

    for ch_i, ch_name in zip(cc, alphabet):
        chain_letters[list(ch_i)] = ch_name

    return chain_letters

@dataclass
class Indep:
    seq: torch.Tensor # [L]
    xyz: torch.Tensor # [L, 36?, 3]
    idx: torch.Tensor

    # SM specific
    bond_feats: torch.Tensor
    chirals: torch.Tensor
    atom_frames: torch.Tensor
    same_chain: torch.Tensor
    is_sm: torch.Tensor
    terminus_type: torch.Tensor

    def write_pdb(self, path, **kwargs):
        with open(path, kwargs.pop('file_mode', 'w')) as fh:
            return self.write_pdb_file(fh, **kwargs)
    
    def write_pdb_file(self, fh, **kwargs):
        seq = self.seq
        seq = torch.where(seq == 20, 0, seq)
        seq = torch.where(seq == 21, 0, seq)
        chain_letters = self.chains()
        return pdbio.writepdb_file(fh,
            torch.nan_to_num(self.xyz[:,:14]), seq, idx_pdb=self.idx, chain_letters=chain_letters, bond_feats=self.bond_feats[None], **kwargs)
    
    def chains(self):
        return chain_letters_from_same_chain(self.same_chain)

@dataclass
class RFI:
    msa_latent: torch.Tensor
    msa_full: torch.Tensor
    seq: torch.Tensor
    seq_unmasked: torch.Tensor
    xyz: torch.Tensor
    sctors: torch.Tensor
    idx: torch.Tensor
    bond_feats: torch.Tensor
    dist_matrix: torch.Tensor
    chirals: torch.Tensor
    atom_frames: torch.Tensor
    t1d: torch.Tensor
    t2d: torch.Tensor
    xyz_t: torch.Tensor
    alpha_t: torch.Tensor
    mask_t: torch.Tensor
    same_chain: torch.Tensor
    is_motif: torch.Tensor
    msa_prev: torch.Tensor
    pair_prev: torch.Tensor
    state_prev: torch.Tensor

@dataclass
class RFO:
    logits: torch.Tensor      # ([1, 61, L, L], [1, 61, L, L], [1, 37, L, L], [1, 19, L, L])
    logits_aa: torch.Tensor   # [1, 80, 115]
    logits_pae: torch.Tensor  # [1, 64, L, L]
    logits_pde: torch.Tensor  # [1, 64, L, L]
    p_bind: torch.Tensor      # [1,1]
    xyz: torch.Tensor         # [40, 1, L, 3, 3]
    alpha_s: torch.Tensor     # [40, 1, L, 20, 2]
    xyz_allatom: torch.Tensor # [1, L, 36, 3]
    lddt: torch.Tensor        # [1, 50, L]
    msa: torch.Tensor
    pair: torch.Tensor
    state: torch.Tensor

    # dataclass.astuple returns a deepcopy of the dataclass in which
    # gradients of member tensors are detached, so we define a 
    # custom unpacker here.
    def unsafe_astuple(self):
        return tuple([self.__dict__[field.name] for field in dataclasses.fields(self)])

    def get_seq_logits(self):
        return self.logits_aa.permute(0,2,1)
    
    def get_xyz(self):
        return self.xyz_allatom[0]

def filter_het(pdb_lines, ligand):
    lines = []
    hetatm_ids = []
    for l in pdb_lines:
        if 'HETATM' not in l:
            continue
        if l[17:17+4].strip() != ligand:
            continue
        lines.append(l)
        hetatm_ids.append(int(l[7:7+5].strip()))

    violations = []
    for l in pdb_lines:
        if 'CONECT' not in l:
            continue
        ids = [int(e.strip()) for e in l[6:].split()]
        if all(i in hetatm_ids for i in ids):
            lines.append(l)
            continue
        if any(i in hetatm_ids for i in ids):
            ligand_atms_bonded_to_protein = [i for i in ids if i in hetatm_ids]
            violations.append(f'line {l} references atom ids in the target ligand {ligand}: {ligand_atms_bonded_to_protein} and another atom')
    if violations:
        raise Exception('\n'.join(violations))
    return lines


def make_indep(pdb, ligand=None, center=True):
    chirals = torch.Tensor()
    atom_frames = torch.zeros((0,3,2))

    xyz_prot, mask_prot, idx_prot, seq_prot = parse_pdb(pdb, seq=True)

    target_feats = inference.utils.parse_pdb(pdb)
    xyz_prot, mask_prot, idx_prot, seq_prot = target_feats['xyz'], target_feats['mask'], target_feats['idx'], target_feats['seq']
    xyz_prot[:,14:] = 0 # remove hydrogens
    mask_prot[:,14:] = False
    xyz_prot = torch.tensor(xyz_prot)
    mask_prot = torch.tensor(mask_prot)
    xyz_prot[~mask_prot] = np.nan
    protein_L, nprotatoms, _ = xyz_prot.shape
    msa_prot = torch.tensor(seq_prot)[None].long()
    if ligand:
        with open(pdb, 'r') as fh:
            stream = [l for l in fh if "HETATM" in l or "CONECT" in l]
        stream = filter_het(stream, ligand)
        if not len(stream):
            raise Exception(f'ligand {ligand} not found in pdb: {pdb}')

        mol, msa_sm, ins_sm, xyz_sm, _ = parse_mol("".join(stream), filetype="pdb", string=True)
        G = rf2aa.util.get_nxgraph(mol)
        atom_frames = rf2aa.util.get_atom_frames(msa_sm, G)
        N_symmetry, sm_L, _ = xyz_sm.shape
        Ls = [protein_L, sm_L]
        msa = torch.cat([msa_prot[0], msa_sm])[None]
        chirals = get_chirals(mol, xyz_sm[0])
        if chirals.numel() !=0:
            chirals[:,:-1] += protein_L
    else:
        Ls = [msa_prot.shape[-1], 0]
        N_symmetry = 1
        msa = msa_prot

    xyz = torch.full((N_symmetry, sum(Ls), ChemData().NTOTAL, 3), np.nan).float()
    mask = torch.full(xyz.shape[:-1], False).bool()
    xyz[:, :Ls[0], :nprotatoms, :] = xyz_prot.expand(N_symmetry, Ls[0], nprotatoms, 3)
    if ligand:
        xyz[:, Ls[0]:, 1, :] = xyz_sm
    xyz = xyz[0]
    mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, Ls[0], nprotatoms)
    idx_sm = torch.arange(max(idx_prot),max(idx_prot)+Ls[1])+200
    idx_pdb = torch.concat([torch.tensor(idx_prot), idx_sm])
    
    seq = msa[0]
    
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    bond_feats[:Ls[0], :Ls[0]] = rf2aa.util.get_protein_bond_feats(Ls[0])
    if ligand:
        bond_feats[Ls[0]:, Ls[0]:] = rf2aa.util.get_bond_feats(mol)


    same_chain = torch.zeros((sum(Ls), sum(Ls))).long()
    same_chain[:Ls[0], :Ls[0]] = 1
    same_chain[Ls[0]:, Ls[0]:] = 1
    is_sm = torch.zeros(sum(Ls)).bool()
    is_sm[Ls[0]:] = True
    assert len(Ls) <= 2, 'multi chain inference not implemented yet'
    terminus_type = torch.zeros(sum(Ls))
    terminus_type[0] = N_TERMINUS
    terminus_type[Ls[0]-1] = C_TERMINUS

    if center:
        xyz = get_init_xyz(xyz[None, None], is_sm).squeeze()
    xyz[is_sm, 0] = 0
    xyz[is_sm, 2] = 0
    indep = Indep(
        seq,
        xyz,
        idx_pdb,
        # SM specific
        bond_feats,
        chirals,
        atom_frames,
        same_chain,
        is_sm,
        terminus_type)
    return indep

class Model:

    def __init__(self, conf):
        self.conf = conf
        self.NTOKENS = ChemData().NAATOKENS
        self.atomizer = None
        self.converter = XYZConverter()

    def forward(self, rfi, **kwargs):
        rfi_dict = dataclasses.asdict(rfi)
        a = self.model(**{**rfi_dict, **kwargs})
        return RFO(*a)

    def insert_contig(self, indep, contig_map, partial_T=False):
        o = copy.deepcopy(indep)

        # Insert small mol into contig_map
        all_chains = set(ch for ch,_ in contig_map.hal)
        # Not yet implemented due to index shifting
        assert_that(len(all_chains)).is_equal_to(1)
        next_unused_chain = next(e for e in contig_map.chain_order if e not in all_chains)
        n_sm = indep.is_sm.sum()
        is_sm_idx0 = torch.nonzero(indep.is_sm, as_tuple=True)[0].tolist()
        contig_map.ref_idx0.extend(is_sm_idx0)
        n_protein_hal = len(contig_map.hal)
        contig_map.hal_idx0 = np.concatenate((contig_map.hal_idx0, np.arange(n_protein_hal, n_protein_hal+n_sm)))
        max_hal_idx = max(i for _, i  in contig_map.hal)
        contig_map.hal.extend(zip([next_unused_chain]*n_sm, range(max_hal_idx+200,max_hal_idx+200+n_sm)))
        chain_id = np.array([c for c, _ in contig_map.hal])
        L_mapped = len(contig_map.hal)
        n_prot = L_mapped - n_sm
        L_in, NATOMS, _ = indep.xyz.shape
        o.xyz = torch.full((L_mapped, NATOMS, 3), np.nan)

        o.xyz[contig_map.hal_idx0] = indep.xyz[contig_map.ref_idx0]
        o.seq = torch.full((L_mapped,), ChemData().MASKINDEX)
        o.seq[contig_map.hal_idx0] = indep.seq[contig_map.ref_idx0]
        o.is_sm = torch.full((L_mapped,), 0).bool()
        o.is_sm[contig_map.hal_idx0] = indep.is_sm[contig_map.ref_idx0]
        o.same_chain = torch.tensor(chain_id[None, :] == chain_id[:, None])
        o.xyz = get_init_xyz(o.xyz[None, None], o.is_sm).squeeze()

        o.bond_feats = torch.full((L_mapped, L_mapped), 0).long()
        o.bond_feats[:n_prot, :n_prot] = rf2aa.util.get_protein_bond_feats(n_prot)
        n_prot_ref = L_in-n_sm
        o.bond_feats[n_prot:, n_prot:] = indep.bond_feats[n_prot_ref:, n_prot_ref:]

        hal_by_ref_d = dict(zip(contig_map.ref_idx0, contig_map.hal_idx0))
        def hal_by_ref(ref):
            return hal_by_ref_d[ref]
        hal_by_ref = np.vectorize(hal_by_ref, otypes=[float])
        o.chirals[...,:-1] = torch.tensor(hal_by_ref(o.chirals[...,:-1]))

        o.idx = torch.tensor([i for _, i in contig_map.hal])

        o.terminus_type = torch.zeros(L_mapped)
        o.terminus_type[0] = N_TERMINUS
        o.terminus_type[n_prot-1] = C_TERMINUS

        is_diffused_prot = ~torch.from_numpy(contig_map.inpaint_str)
        is_diffused_sm = torch.zeros(n_sm).bool()
        is_diffused = torch.cat((is_diffused_prot, is_diffused_sm))
        is_atom_str_shown = contig_map.atomize_indices2atomname
        # The motifs for atomization are double-counted.
        if is_atom_str_shown:
            is_diffused[list(is_atom_str_shown.keys())] = True
        o = copy.deepcopy(o)
        self.atomizer = None

        # ComputeAllAtom in the network requires N and C coords even for atomized residues,
	    # however these have no semantic value.
        sm_ca = o.xyz[o.is_sm, 1]
        o.xyz[o.is_sm,:3] = sm_ca[...,None,:]
        o.xyz[o.is_sm] += chemical.INIT_CRDS

        return o, is_diffused, is_diffused


    def prepro(self, indep, t, is_diffused):
        """
        Function to prepare inputs to diffusion model.

        Prepared in a form recognizable by the structure prediction network to
        enable finetuning.
        """

        xyz_t = indep.xyz
        seq_one_hot = torch.nn.functional.one_hot(
                indep.seq, num_classes=self.NTOKENS).float()
        L = seq_one_hot.shape[0]


        # msa_full:   NSEQ,NINDEL,NTERMINUS,
        # msa_masked: NSEQ,NSEQ,NINDEL,NINDEL,NTERMINUS
        NTERMINUS = 2
        NINDEL = 1

        ### msa_masked ###
        ##################
        msa_masked = torch.zeros((1,1,L,2*ChemData().NAATOKENS+NINDEL*2+NTERMINUS))

        msa_masked[:,:,:,:ChemData().NAATOKENS] = seq_one_hot[None, None]
        msa_masked[:,:,:,ChemData().NAATOKENS:2*ChemData().NAATOKENS] = seq_one_hot[None, None]
        msa_masked[:,:,:,MSAMASKED_N_TERM] = (indep.terminus_type == N_TERMINUS).float()
        msa_masked[:,:,:,MSAMASKED_C_TERM] = (indep.terminus_type == C_TERMINUS).float()

        ### msa_full ###
        ################
        msa_full = torch.zeros((1,1,L,ChemData().NAATOKENS+NINDEL+NTERMINUS))
        msa_full[:,:,:,:ChemData().NAATOKENS] = seq_one_hot[None, None]
        msa_full[:,:,:,MSAFULL_N_TERM] = (indep.terminus_type == N_TERMINUS).float()
        msa_full[:,:,:,MSAFULL_C_TERM] = (indep.terminus_type == C_TERMINUS).float()

        ### t1d ###
        ########### 
        # Here we need to go from one hot with 22 classes to one hot with 21 classes
        # If sequence is masked, it becomes unknown
        seq_cat_shifted = seq_one_hot.argmax(dim=-1)
        seq_cat_shifted[seq_cat_shifted>=ChemData().MASKINDEX] -= 1
        t1d = torch.nn.functional.one_hot(seq_cat_shifted, num_classes=ChemData().NAATOKENS-1)
        t1d = t1d[None, None] # [L, ChemData().NAATOKENS-1] --> [1,1,L, NAATOKENS-1]
        
        # Set confidence to 1 where diffusion mask is True, else 1-t/T
        strconf = torch.zeros((L,)).float()
        strconf[~is_diffused] = 1.
        strconf[is_diffused] = 1. - t/self.conf.diffuser.T
        strconf = strconf[None,None,...,None]

        t1d = torch.cat((t1d, strconf), dim=-1)
        t1d = t1d.float()

        xyz_t[is_diffused,3:,:] = float('nan')

        assert_that(xyz_t.shape).is_equal_to((L,ChemData().NHEAVYPROT,3))
        xyz_t=xyz_t[None, None]
        xyz_t = torch.cat((xyz_t, torch.full((1,1,L,ChemData().NTOTAL-ChemData().NHEAVYPROT,3), float('nan'))), dim=3)

        t2d = None
        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)

        alpha, _, alpha_mask, _ = self.converter.get_torsions(xyz_t.reshape(-1,L,ChemData().NTOTAL,3), seq_tmp)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(-1,L,ChemData().NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(-1,L,ChemData().NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(-1, L, 3*ChemData().NTOTALDOFS) # [n,L,30]

        alpha_t = alpha_t.unsqueeze(1) # [n,I,L,30]
        
        mask_t = torch.ones(1,1,L,L).bool()
        sctors = torch.zeros((1,L,ChemData().NTOTALDOFS,2))

        xyz = torch.squeeze(xyz_t, dim=0)

        # NO SELF COND
        xyz_t = torch.zeros(1,1,L,3)
        t2d = torch.zeros(1,1,L,L,68)

        is_protein_motif = ~is_diffused * ~indep.is_sm
        xyz[0, is_diffused*~indep.is_sm,3:] = torch.nan
        xyz[0, indep.is_sm,14:] = 0
        xyz[0, is_protein_motif, 14:] = 0

        dist_matrix = get_bond_distances(indep.bond_feats)

        

        # minor tweaks to rfi to match gp training
        if ('inference' in self.conf) and (self.conf.inference.get('contig_as_guidepost', False)):
            '''Manually inspecting the pickled features passed to RF during training, 
            I did not see markers for the N and C termini. This is to more accurately 
            replicate the features seen during training at inference.'''
            # Erase N/C termini markers
            msa_masked[...,-2:] = 0
            msa_full[...,-2:] = 0

        # Note: should be batched
        rfi = RFI(
            msa_masked,
            msa_full,
            indep.seq[None],
            indep.seq[None],
            xyz,
            sctors,
            indep.idx[None],
            indep.bond_feats[None],
            dist_matrix[None],
            indep.chirals[None],
            indep.atom_frames[None],
            t1d,
            t2d,
            xyz_t,
            alpha_t,
            mask_t,
            indep.same_chain[None],
            ~is_diffused,
            None,
            None,
            None)
        return rfi
    

def assert_has_coords(xyz, indep):
    assert len(xyz.shape) == 3
    missing_backbone = torch.isnan(xyz).any(dim=-1)[...,:3].any(dim=-1)
    prot_missing_bb = missing_backbone[~indep.is_sm]
    sm_missing_ca = torch.isnan(xyz).any(dim=-1)[...,1]
    try:
        assert not prot_missing_bb.any(), f'prot_missing_bb {prot_missing_bb}'
        assert not sm_missing_ca.any(), f'sm_missing_ca {sm_missing_ca}'
    except Exception as e:
        print(e)
        import ipdb
        ipdb.set_trace()

def pad_dim(x, dim, new_l, value=0):
    padding = [0]*2*x.ndim
    padding[2*dim] = new_l - x.shape[dim]
    padding = padding[::-1]
    return F.pad(x, pad=tuple(padding), value=value)

def write_traj(path, xyz_stack, seq, bond_feats, natoms=23, **kwargs):
    xyz23 = pad_dim(xyz_stack, 2, natoms)
    if bond_feats is not None:
        bond_feats = bond_feats[None]
    with open(path, 'w') as fh:
        for i, xyz in enumerate(xyz23):
            rf2aa.util.writepdb_file(fh, xyz, seq, bond_feats=bond_feats, modelnum=i, **kwargs)

def forward(model, rfi, **kwargs):
    rfi_dict = dataclasses.asdict(rfi)
    return RFO(*model(**{**rfi_dict, **kwargs}))

def mask_indep(indep, is_diffused):
    indep.seq[is_diffused] = ChemData().MASKINDEX

def self_cond(indep, rfi, rfo):
    # RFI is already batched
    B = 1
    L = indep.xyz.shape[0]
    rfi_sc = copy.deepcopy(rfi)
    zeros = torch.zeros(B,1,L,36-3,3).float().to(rfi.xyz.device)
    xyz_t = torch.cat((rfo.xyz[-1:], zeros), dim=-2) # [B,T,L,27,3]
    t2d, mask_t_2d_remade = util.get_t2d(
        xyz_t[0], indep.is_sm, rfi.atom_frames[0])
    t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
    rfi_sc.xyz_t = xyz_t[:,:,:,1]
    rfi_sc.t2d = t2d
    return rfi_sc

def hetatm_names(pdb):
    d = defaultdict(list)
    with open(pdb) as f:
        for line in f.readlines():
            if line.startswith('HETATM'):
                lig_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                element_name = line[76:78].strip()
                d[lig_name].append((atom_name, element_name))
    return d

def without_H(atom_elem_by_lig):
    ''' Drops Hs from a dictionary like {'LG1': [('CB', 'C'), ('H2', 'H')]}'''
    out = {}
    for lig, atom_names in atom_elem_by_lig.items():
        out[lig] = [(atom_name, element) for atom_name, element in atom_names if element != 'H']
    return out

def rename_ligand_atoms(ref_fn, out_fn):
    """Copies names of ligand residue and ligand heavy atoms from input pdb
    into output (design) pdb."""

    ref_atom_names_by_lig = hetatm_names(ref_fn)
    ref_atom_names_by_lig = without_H(ref_atom_names_by_lig)
    with open(out_fn) as f:
        lines = [line.strip() for line in f.readlines()]

    lines2 = []
    ligand_counters = defaultdict(lambda: 0)
    for line in lines:
        if line.startswith('HETATM'):
            lig_name = line[17:20].strip()
            element_name = line[76:78].strip()
            assertpy.assert_that(ref_atom_names_by_lig).contains(lig_name)
            assertpy.assert_that(element_name).is_not_equal_to('H')
            ref_atom_name, ref_element_name = ref_atom_names_by_lig[lig_name][ligand_counters[lig_name]]
            assertpy.assert_that(element_name.upper()).is_equal_to(ref_element_name.upper())
            ligand_counters[lig_name] += 1
            line = line[:12] + ref_atom_name.ljust(4, ' ') + line[16:]
            line = line[:76] + ref_element_name.rjust(2, ' ') + line[78:]
        if line.startswith('MODEL'):
            ligand_counters = defaultdict(lambda: 0)
        lines2.append(line)

    with open(out_fn,'w') as f:
        for line in lines2:
            print(line, file=f)
