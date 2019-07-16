import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedulers
import numpy as np
from collections import defaultdict

class SET_zero(SparseModel):


    def find_first_pos(self, tensor, value):
        idx = torch.argmin(torch.abs(tensor-value)).item()
        return idx

    def find_last_pos(self, tensor, value):
        """ 
        Invert tensor reference [1]: 
        [1] https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382

        """
        idx = [i for i in range(tensor.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx).to(self.device)
        inv_tensor = tensor.index_select(0, idx)
        idx = torch.argmin(torch.abs(inv_tensor-value)).item()
        return tensor.shape[0] - idx

    def _run_one_pass(self, loader, train):
        super(SET_zero, self)._run_one_pass(loader, train)
        if train:
            self.reinitialize_weights()

    def prune(self, A, num_params, zeta=0.3):
        """ Calculate new weights based on SET approach 
            Implementation follows exact same steps in original repo [1]

            [1] https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-
            networks/blob/master/SET-MLP-Keras-Weights-Mask/set_mlp_keras_cifar10.py
        """

        with torch.no_grad():
            values, _ = torch.sort(A.view(-1))
            first_zero_pos = self.find_first_pos(values, 0)
            last_zero_pos = self.find_last_pos(values, 0)
            len_values = values.shape[0]

            largest_negative = values[int((1-zeta)*first_zero_pos)]
            smallest_positive = values[
                int(min(len_values-1, 
                    last_zero_pos + zeta * (len_values - last_zero_pos)))
            ]

            # create new array (easier to initialize all to zero)
            R = A.clone()
            R[R > smallest_positive] = 1
            R[R < largest_negative] = 1
            R[R != 1] = 0
            W = R.clone()

            # add random weights
            num_add = 0
            num_rewires = num_params - torch.sum(R).item()
            while (num_add < num_rewires):
                i = np.random.randint(0, R.shape[0])
                j = np.random.randint(0, R.shape[1])
                if (R[i,j] == 0):
                    R[i,j] = 1
                    num_add += 1

        # R is the rewired weights, or the new mask
        # W is the mask with just the previous values (the prune maskl)
        return R, W

    def reinitialize_weights(self):        
        """ Reinitialize weights """

        layers = iter(range(len(self.masks)))
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                idx = next(layers, None)
                if idx is not None:
                    layer_weights = m.weight.clone().detach()                
                    new_mask, prune_mask = self.prune(layer_weights, self.num_params[idx])
                    with torch.no_grad():
                        self.masks[idx] = new_mask.float()
                        m.weight.data *= prune_mask.float()
                        # keep track of mean and std of weights
                        self.log['layer_' + str(idx) + '_mean'] = torch.mean(m.weight).item()
                        self.log['layer_' + str(idx) + '_std'] = torch.std(m.weight).item()

        # keep track of mask sizes when debugging
        if self.debug_sparse:
            for idx, m in enumerate(self.masks):    
                self.log['mask_sizes_l' + str(idx+1)] = torch.sum(m).item()


class SET_sameDist(SparseModel):

    def prune(self, A, M, zeta=0.3):
            """ Calculate new weights based on SET approach 
                
                Arguments:
                - A: current weight matrix
                - M: mask
                - W: original weights
            """

            with torch.no_grad():

                # calculate thresholds and decay weak connections
                A_pos = A[A>0]
                pos_threshold, _ = torch.kthvalue(A_pos, int(zeta*len(A_pos)))
                A_neg = A[A<0]
                neg_threshold, _ = torch.kthvalue(A_neg, int((1-zeta)*len(A_neg)))
                N = ((A >= pos_threshold) | (A <= neg_threshold)).to(self.device)

                # randomly select new connections, zero out conns which had previous weights
                gamma = 1.00
                shape = A.shape
                on_perc = torch.sum(A != 0).item() / np.prod(shape)
                p_update = zeta * on_perc * gamma / (1-on_perc)
                P = torch.rand(shape).to(self.device) < p_update        
                M_prime = N | (P & (M==0))
                
                # get new weights matrix
                Z = (torch.randn(A.shape) * 1e-2).to(self.device)
                Z = Z * (P & (M==0)).float()
            
            # return new weights and mask 
            return M_prime, Z

    def reinitialize_weights(self):        
        """ Reinitialize weights """

        layers = iter(range(len(self.masks)))
        for m in self.network.modules():
            if isinstance(m, nn.Linear):           
                idx = next(layers, None)
                if idx is not None:                
                    new_mask, new_weight = self.prune(m.weight, self.masks[idx])
                    self.masks[idx] = new_mask.float()
                    m.weight = torch.nn.Parameter((m.weight + new_weight) * self.masks[idx])
                    # keep track of mean and std of weights
                    self.log['layer_' + str(idx) + '_mean'] = torch.mean(m.weight).item()
                    self.log['layer_' + str(idx) + '_std'] = torch.std(m.weight).item()

        # keep track of mask sizes when debugging
        if self.debug_sparse:
            self.log['mask_sizes'] = [torch.sum(m) for m in self.masks].tolist()

                
