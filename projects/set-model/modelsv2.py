import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BaseModel():

    def __init__(self, network, config={}):

        defaults = dict(
            optim_alg='SGD', 
            device='cpu', 
        )
        defaults.update(config)
        self.__dict__.update(defaults)

        # init remaining
        self.device = torch.device(self.device)
        self.network = network.to(self.device)

    def setup(self, init_weights=True):

        # init optimizer
        if self.optim_alg == 'Adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)
        elif self.optim_alg == 'SGD':
            self.optimizer = optim.SGD(self.network.parameters(), lr=1e-2, 
                                       weight_decay=2e-4, momentum=0.9)
        # init loss function
        self.loss_func = nn.CrossEntropyLoss()

    def run_epoch(self, dataset):
        self.network.train()
        self.log={}
        self.log['train_loss'], self.log['train_acc'] = self._run_one_pass(
            dataset.train_loader, train=True)
        self.network.eval()
        self.log['val_loss'], self.log['val_acc'] = self._run_one_pass(
            dataset.test_loader, train=False)
        return self.log

    def _run_one_pass(self, loader, train):
        epoch_loss = 0
        num_samples = 0
        correct = 0
        for data in loader:
            # get inputs and label
            inputs, targets = data
            targets = targets.to(self.device)

            # zero gradients
            if train:
                self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.network(inputs)
            correct += (targets == torch.max(outputs, dim=1)[1]).sum().item()
            loss = self.loss_func(outputs, targets)
            if train:
                loss.backward()
                self.optimizer.step()

            # keep track of loss
            epoch_loss += loss.item()
            num_samples += inputs.shape[0]

        return (epoch_loss / num_samples) * 1000, correct / num_samples

    def save(self):
        pass

    def restore(self):
        pass


class SparseModel(BaseModel):

    def setup(self, init_weights=True, epsilon=20):
        super(SparseModel, self).setup(init_weights)

        # calculate sparsity masks
        self.masks = []
        self.denseness = []
        linear_layers = [m for m in self.network.modules() if isinstance(m, nn.Linear)]
        # don't need a mask for the last layer
        for layer in linear_layers[:-1]:
            shape = layer.weight.shape
            on_perc = epsilon * np.sum(shape)/np.prod(shape)
            mask = (torch.rand(shape) < on_perc).float().to(self.device)
            layer.weight = torch.nn.Parameter(layer.weight * mask)
            self.masks.append(mask)
            self.denseness.append(on_perc)

    def _run_one_pass(self, loader, train):
        epoch_loss = 0
        num_samples = 0
        correct = 0
        for data in loader:
            # get inputs and label
            inputs, targets = data
            targets = targets.to(self.device)

            # zero gradients
            if train:
                self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.network(inputs)
            correct += (targets == torch.max(outputs, dim=1)[1]).sum().item()
            loss = self.loss_func(outputs, targets)
            if train:
                loss.backward()
                # zero the gradients for dead connections
                masks = iter(self.masks)
                for m in self.network.modules():
                    if isinstance(m, nn.Linear):
                        mask = next(masks, None)
                        if mask is not None:
                            m.weight.grad *= mask
                self.optimizer.step()

            # keep track of loss
            epoch_loss += loss.item()
            num_samples += inputs.shape[0]

        # add monitoring of sparse levels
        if train and self.debug_sparse: 
            self._log_sparse_levels()

        return (epoch_loss / num_samples) * 1000, correct / num_samples

    def _log_sparse_levels(self):
        self.log['sparse_levels'] = []
        layers = iter(range(len(self.masks)))
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                idx = next(layers, None)
                if idx is not None: 
                    zeros = torch.sum((m.weight == 0).int()).item()
                    size = np.prod(m.weight.shape)
                    self.log['sparse_levels'].append((1- zeros/size))

class SET_zero(SparseModel):

    def prune(self, A, M, perc_on, zeta=0.3):
            """ Calculate new weights based on SET approach 
                
                Arguments:
                - A: current weight matrix
                - M: mask
                - W: original weights
            """

            with torch.no_grad():
                
                # reduce the number of connections to be pruned by the number 
                # of those which are already 0, so sparsity is not increased
                shape = A.shape
                on_weights = A[M.byte()]
                count_zeros = torch.sum(on_weights==0).item()
                n_prune = zeta - count_zeros/np.prod(shape)
                print("count zeros", count_zeros)
                print("n prune: ", n_prune)
                               
                # calculate thresholds and decay weak connections
                A_pos = A[A>0]
                pos_threshold, _ = torch.kthvalue(A_pos, int(n_prune*len(A_pos)))
                A_neg = A[A<0]
                neg_threshold, _ = torch.kthvalue(A_neg, int((1-n_prune)*len(A_neg)))
                N = ((A >= pos_threshold) | (A <= neg_threshold)).to(self.device)
                print("N: ", torch.sum(N))

                # randomly select new connections, zero out conns which had previous weights
                gamma = 1.00
                print("perc_on: ", perc_on)
                p_update = zeta * perc_on * gamma / (1-perc_on)
                P = torch.rand(shape).to(self.device) < p_update        
                M_prime = N | (P & (M==0))
            
            # return new weights and mask 
            return M_prime

    def reinitialize_weights(self):        
        """ Reinitialize weights """

        layers = iter(range(len(self.masks)))
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                idx = next(layers, None)
                if idx is not None:                
                    new_mask = self.prune(m.weight, self.masks[idx], self.denseness[idx])
                    self.masks[idx] = new_mask.float()
                    m.weight = torch.nn.Parameter(m.weight * self.masks[idx])
                    # keep track of mean and std of weights
                    self.log['layer_' + str(idx) + '_mean'] = torch.mean(m.weight).item()
                    self.log['layer_' + str(idx) + '_std'] = torch.std(m.weight).item()

        # keep track of mask sizes when debugging
        if self.debug_sparse:
            self.log['mask_sizes'] = [torch.sum(m) for m in self.masks].tolist()

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


