import os
import glob
from icecream import ic
from tqdm import tqdm
import fire
import aa_model
import torch
import rf2aa.util


def get_ligands(pdb_lines):
    ligands = set()
    for l in pdb_lines:
        if 'HETATM' not in l:
            continue
        curr_ligand = l[17:17+4].strip()
        ligands.add(curr_ligand)
    return ligands

def rewrite(path, outpath):
    with open(path, 'r') as fh:
        stream = [l for l in fh if "HETATM" in l or "CONECT" in l]

    ligands = get_ligands(stream)
    indep = aa_model.make_indep(path, ','.join(ligands), center=False)
    xyz = indep.xyz[~indep.is_sm]
    idx = indep.idx[~indep.is_sm]
    L = xyz.shape[0]
    ala_seq = torch.zeros((L,))
    xyz = rf2aa.util.idealize_reference_frame(ala_seq[None], xyz[None])[0]
    xyz_ideal = get_o(xyz, idx)
    indep.xyz[~indep.is_sm, :4] = xyz_ideal
    ligands = list(ligands)
    assert len(ligands) == 1, f'Found >1 ligand: {ligands}'
    indep.write_pdb(outpath, lig_name=ligands[0])

def get_o(xyz, idx):
    idx_pad = torch.concat([idx, torch.tensor([-1])])
    is_adj = (idx_pad[:-1] - idx_pad[1:]) == -1
    L = xyz.shape[0]
    xyz_ideal = torch.zeros((L, 4, 3))
    xyz_ideal[:,:3] = xyz[:,:3]
    for frames, idxs, ideal_pos in [
        (
            (xyz[:,0,:],xyz[:,1,:],xyz[:,2,:]),
            torch.nonzero(~is_adj),
            torch.tensor([2.1428,  0.7350, -0.7413]),
        ),
        (
            (xyz[:-1,1,:],xyz[:-1,2,:],xyz[1:,0,:]),
            torch.nonzero(is_adj),
            torch.tensor([  -0.7247,     -1.0032,     -0.0003])
        )]:
        idxs = idxs[:,0]
    
        Rs, Ts = rf2aa.util.rigid_from_3_points(frames[0], frames[1], frames[2])
        Rs = Rs[idxs]
        Ts = Ts[idxs]

        xyz_ideal[idxs, 3] = torch.einsum('lij,j->li', Rs, ideal_pos) + Ts

    return xyz_ideal

def main(pattern, outdir):
    for pdb in tqdm(sorted(glob.glob(pattern))):
        d, name = os.path.split(pdb)
        outpath = os.path.join(outdir, name)
        ic(name, pdb, outpath)
        rewrite(pdb, outpath)

if __name__ == '__main__':
    fire.Fire(main)
