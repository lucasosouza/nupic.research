import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import logging
print = logging.info

class MLP(nn.Module):
    
    def __init__(self, input_size=784, num_classes=10, init_weights=True,
                 hidden_sizes = [1000, 1000, 1000], device=None):
        
        super(MLP, self).__init__()
        self.device = device
        self.input_size = input_size
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], num_classes)
        ).to(self.device)
        
        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        return self.classifier(x.view(-1, self.input_size).to(self.device))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class MLP_dense(nn.Module):
    """ Added:
        - Relu
        - batchnorm
        - dropout at second to last layer
        - weight decay """
    
    def __init__(self, input_size=784, num_classes=10, init_weights=True,
                 hidden_sizes = [1000, 1000, 1000], device=None, bias=False):
        
        super(MLP_dense, self).__init__()
        self.device = device
        self.input_size = input_size

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, hidden_sizes[0], bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Linear(hidden_sizes[1], hidden_sizes[2], bias=bias),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Linear(hidden_sizes[2], num_classes, bias=bias)
        ).to(self.device)

        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        return self.classifier(x.view(-1, self.input_size).to(self.device))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    
class MLP_sparse(nn.Module):
    """ Added:
        - Sparse connections
    """
    
    def __init__(self, input_size=784, num_classes=10, init_weights=True,
                 hidden_sizes = [1000, 1000, 1000], device=None, bias=False,
                 epsilon=20):

        super(MLP_sparse, self).__init__()
        self.device = device
        self.input_size = input_size

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, hidden_sizes[0], bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Linear(hidden_sizes[1], hidden_sizes[2], bias=bias),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Linear(hidden_sizes[2], num_classes, bias=bias)
        ).to(self.device)

        # calculate sparsity masks
        self.masks = []
        linear_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        # don't need a mask for the last layer
        for layer in linear_layers[:-1]:
            shape = layer.weight.shape
            sparsity = epsilon * np.sum(shape)/np.prod(shape)
            mask = torch.rand(shape) < sparsity
            self.masks.append(mask.float().to(self.device))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        return self.classifier(x.view(-1, self.input_size).to(self.device))

    def _initialize_weights(self):
        masks = iter(self.masks)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):           
                with torch.no_grad():
                    # initialize weights
                    weight = (torch.randn(m.weight.shape) * 1e-2).to(self.device)
                    mask = next(masks, None)
                    if mask is not None:
                        m.weight = torch.nn.Parameter(weight * mask)
                    else:
                        m.weight = torch.nn.Parameter(weight)
    
    def print_sparse_levels(self):
        print("------------------------")
        layers = iter(range(len(self.masks)))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                idx = next(layers, None)
                if idx is not None:                
                    zeros = torch.sum((m.weight == 0).int()).item()
                    size = np.prod(m.weight.shape)
                    print(1- zeros/size)
        print("------------------------")
    

class MLP_set(nn.Module):
    """ Added:
        - SEP Pruning
    """
    
    def __init__(self, input_size=784, num_classes=10, init_weights=True,
                 hidden_sizes = [1000, 1000, 1000], device=None, bias=False,
                 epsilon=20):

        super(MLP_set, self).__init__()
        self.device = device
        self.input_size = input_size

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, hidden_sizes[0], bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Linear(hidden_sizes[1], hidden_sizes[2], bias=bias),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Linear(hidden_sizes[2], num_classes, bias=bias)
        ).to(self.device)

        # calculate sparsity masks
        self.masks = []
        linear_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        # don't need a mask for the last layer
        for layer in linear_layers[:-1]:
            shape = layer.weight.shape
            sparsity = epsilon * np.sum(shape)/np.prod(shape)
            mask = torch.rand(shape) < sparsity
            self.masks.append(mask.float().to(self.device))
            
        # keep track of dead weights
        self.weights = []

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        return self.classifier(x.view(-1, self.input_size).to(self.device))
   
    def _initialize_weights(self):
        masks = iter(self.masks)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):           
                with torch.no_grad():
                    # initialize weights
                    weight = (torch.randn(m.weight.shape) * 1e-2).to(self.device)
                    # mask weights if there is a mask
                    mask = next(masks, None)
                    if mask is not None:
                        self.weights.append(weight)
                        masked_weight = weight * mask
                        m.weight = torch.nn.Parameter(masked_weight)
                    else:
                        m.weight = torch.nn.Parameter(weight)

    def prune(self, A, M, W, zeta=0.3):
            """ Calculate new weights based on SET approach 
                
                Arguments:
                - A: current weight matrix
                - M: mask
                - W: original weights
            """

            with torch.no_grad():

                # update original weights matrices with the current weights
                W_prime = A * M + W * (M==0).float()

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
            
            # return new weights and mask 
            return W_prime, M_prime

    def reinitialize_weights(self):        
        """ Reinitialize weights """

        layers = iter(range(len(self.masks)))
        for m in self.modules():
            if isinstance(m, nn.Linear):           
                idx = next(layers, None)
                if idx is not None:                
                    new_weight, new_mask = self.prune(m.weight, self.masks[idx], self.weights[idx])
                    self.weights[idx] = new_weight
                    self.masks[idx] = new_mask.float()
                    m.weight = torch.nn.Parameter(self.weights[idx] * self.masks[idx])
                    # print(torch.mean(m.weight))
                
    def print_sparse_levels(self):
        print("------------------------")
        layers = iter(range(len(self.masks)))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                idx = next(layers, None)
                if idx is not None:                
                    zeros = torch.sum((m.weight == 0).int()).item()
                    size = np.prod(m.weight.shape)
                    print(1- zeros/size)
        print("------------------------")

