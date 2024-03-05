import torch
import rf2aa

class Potential:
    '''
        Interface class that defines the functions a potential must implement
    '''

    def compute(self, seq, xyz):
        '''
            Given the current sequence and structure of the model prediction, return the current
            potential as a PyTorch tensor with a single entry

            Args:
                seq (torch.tensor, size: [L,22]:    The current sequence of the sample.
                xyz (torch.tensor, size: [L,27,3]: The current coordinates of the sample
            
            Returns:
                potential (torch.tensor, size: [1]): A potential whose value will be MAXIMIZED
                                                     by taking a step along it's gradient
        '''
        raise NotImplementedError('Potential compute function was not overwritten')


class ligand_ncontacts(Potential):

    '''
        Differentiable way to maximise number of contacts between binder and target

        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

        Author: PV
    '''


    def __init__(self, weight=1, r_0=8, d_0=4):

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, seq, xyz):

        is_atom = rf2aa.util.is_atom(torch.argmax(seq,dim=1)).cpu()

        # Extract ligand Ca residues
        Ca_l = xyz[is_atom,1] # [Ll,3]

        # Extract binder Ca residues
        Ca_b = xyz[~is_atom,1] # [Lb,3]


        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca_b[None,...].contiguous(), Ca_l[None,...].contiguous(), p=2) # [1,Ll,Lb]
        ligand_ncontacts = -1 * contact_energy(dgram, self.r_0, self.d_0)
        #Potential is the sum of values in the tensor
        ligand_ncontacts = ligand_ncontacts.sum()

        return self.weight * ligand_ncontacts

def contact_energy(dgram, d_0, r_0):
    divide_by_r_0 = (dgram - d_0) / r_0
    numerator = torch.pow(divide_by_r_0,6)
    denominator = torch.pow(divide_by_r_0,12)
    
    ncontacts = (1 - numerator) / ((1 - denominator)).float()
    return - ncontacts


# Dictionary of types of potentials indexed by name of potential. Used by PotentialManager.
# If you implement a new potential you must add it to this dictionary for it to be used by
# the PotentialManager
implemented_potentials = {
                           'ligand_ncontacts':     ligand_ncontacts}

