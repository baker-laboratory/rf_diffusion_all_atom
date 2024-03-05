import torch
import numpy as np

from  rf2aa.chemical import ChemicalData as ChemData

def writepdb(filename, *args, file_mode='w', **kwargs, ):
    f = open(filename, file_mode)
    writepdb_file(f, *args, **kwargs)

def writepdb_file(f, atoms, seq, modelnum=None, chain="A", idx_pdb=None, bfacts=None, 
             bond_feats=None, file_mode="w",atom_mask=None, atom_idx_offset=0, chain_Ls=None,
             remap_atomtype=True, lig_name='LG1', atom_names=None, chain_letters=None):
    #ic(atoms.shape, seq.shape, bond_feats.shape)
    #ic(chain_Ls)

    def _get_atom_type(atom_name):
        atype = ''
        if atom_name[0].isalpha():
            atype += atom_name[0]
        atype += atom_name[1]
        return atype

    # if needed, correct mistake in atomic number assignment in RF2-allatom (fold&dock 3 & earlier)
    atom_names_ = [
        "F",  "Cl", "Br", "I",  "O",  "S",  "Se", "Te", "N",  "P",  "As", "Sb",
        "C",  "Si", "Ge", "Sn", "Pb", "B",  "Al", "Zn", "Hg", "Cu", "Au", "Ni", 
        "Pd", "Pt", "Co", "Rh", "Ir", "Pr", "Fe", "Ru", "Os", "Mn", "Re", "Cr", 
        "Mo", "W",  "V",  "U",  "Tb", "Y",  "Be", "Mg", "Ca", "Li", "K",  "ATM"]
    atom_num = [
        9,    17,   35,   53,   8,    16,   34,   52,   7,    15,   33,   51,
        6,    14,   32,   50,   82,   5,    13,   30,   80,   29,   79,   28,
        46,   78,   27,   45,   77,   59,   26,   44,   76,   25,   75,   24,   
        42,   74,   23,   92,   65,   39,   4,    12,   20,   3,    19,   0] 
    atomnum2atomtype_ = dict(zip(atom_num,atom_names_))
    if remap_atomtype:
        atomtype_map = {v:atomnum2atomtype_[k] for k,v in ChemData().atomnum2atomtype.items()}
    else:
        atomtype_map = {v:v for k,v in ChemData().atomnum2atomtype.items()} # no change
        
    ctr = 1+atom_idx_offset
    scpu = seq.cpu().squeeze(0)
    atomscpu = atoms.cpu().squeeze(0)

    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    assert not (chain_Ls and chain_letters)
    if chain_letters is None:
        if chain_Ls is not None:
            chain_letters = np.concatenate([np.full(L, alphabet[i]) for i,L in enumerate(chain_Ls)])
        else:
            chain_letters = [chain]*len(scpu)
        
    if modelnum is not None:
        f.write(f"MODEL        {modelnum}\n")

    Bfacts = torch.clamp( bfacts.cpu(), 0, 1)
    atom_idxs = {}
    i_res_lig = 0
    for i_res,s,ch in zip(range(len(scpu)), scpu, chain_letters):
        natoms = atomscpu.shape[-2]
        if s >= len(ChemData().aa2long):
            atom_idxs[i_res] = ctr

            # hack to make sure H's are output properly (they are not in RFAA alphabet)
            if atom_names is not None:
                atom_type = _get_atom_type(atom_names[i_res_lig])
                atom_name = atom_names[i_res_lig]
            else:
                atom_type = atomtype_map[ChemData().num2aa[s]]
                atom_name = atom_type

            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %+2s\n"%(
                    "HETATM", ctr, atom_name, lig_name,
                    ch, idx_pdb.max()+10, atomscpu[i_res,1,0], atomscpu[i_res,1,1], atomscpu[i_res,1,2],
                    1.0, Bfacts[i_res],  atom_type) )
            i_res_lig += 1
            ctr += 1
            continue

        atms = ChemData().aa2long[s]

        for i_atm,atm in enumerate(atms):
            if atom_mask is not None and not atom_mask[i_res,i_atm]: continue # skip missing atoms
            if (i_atm<natoms and atm is not None and not torch.isnan(atomscpu[i_res,i_atm,:]).any()):
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, atm, ChemData().num2aa[s],
                    ch, idx_pdb[i_res], atomscpu[i_res,i_atm,0], atomscpu[i_res,i_atm,1], atomscpu[i_res,i_atm,2],
                    1.0, Bfacts[i_res] ) )
                ctr += 1
    if bond_feats != None:
        atom_bonds = (bond_feats > 0) * (bond_feats <5)
        atom_bonds = atom_bonds.cpu()
        b, i, j = atom_bonds.nonzero(as_tuple=True)
        for start, end in zip(i,j):
            f.write(f"CONECT{atom_idxs[int(start.cpu().numpy())]:5d}{atom_idxs[int(end.cpu().numpy())]:5d}\n")
    if modelnum is not None:
        f.write("ENDMDL\n")

