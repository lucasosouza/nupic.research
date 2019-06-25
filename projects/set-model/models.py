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
        self.log={}
        self.network.train()
        self._run_one_pass(dataset.train_loader, train=True)
        self.network.eval()
        self._run_one_pass(dataset.test_loader, train=False)
        return self.log

    def _run_one_pass(self, loader, train=True):
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

        # store loss and acc at each pass
        loss = (epoch_loss / num_samples) * 1000
        acc = correct / num_samples
        if train:
            self.log['train_loss'] = loss
            self.log['train_acc'] = acc
        else:
            self.log['val_loss'] = loss
            self.log['val_acc'] = acc

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
        self.num_params = [] # added for paper implementation
        linear_layers = [m for m in self.network.modules() if isinstance(m, nn.Linear)]
        # don't need a mask for the last layer
        for layer in linear_layers[:-1]:
            shape = layer.weight.shape
            on_perc = epsilon * np.sum(shape)/np.prod(shape)
            mask = (torch.rand(shape) < on_perc).float().to(self.device)
            layer.weight = torch.nn.Parameter(layer.weight * mask)
            self.masks.append(mask)
            self.denseness.append(on_perc)
            self.num_params.append(torch.sum(mask).item())

    def _run_one_pass(self, loader, train):
        """ TODO: reimplement by calling super and passing a hook """ 


        epoch_loss = 0
        num_samples = 0
        correct = 0
        for data in loader:
            # get inputs and label
            inputs, targets = data
            targets = targets.to(self.device)

            # forward + backward + optimize
            self.optimizer.zero_grad()
            outputs = self.network(inputs)
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

            # keep track of loss and accuracy
            correct += (targets == torch.max(outputs, dim=1)[1]).sum().item()
            epoch_loss += loss.item()
            num_samples += inputs.shape[0]

        # store loss and acc at each pass
        loss = (epoch_loss / num_samples) * 1000
        acc = correct / num_samples
        if train:
            self.log['train_loss'] = loss
            self.log['train_acc'] = acc
        else:
            self.log['val_loss'] = loss
            self.log['val_acc'] = acc

        # add monitoring of sparse levels
        if train and self.debug_sparse: 
            self._log_sparse_levels()

    def _log_sparse_levels(self):
        self.log['sparse_levels'] = []
        layers = iter(range(len(self.masks)))
        with torch.no_grad():
            for m in self.network.modules():
                if isinstance(m, nn.Linear):
                    idx = next(layers, None)
                    if idx is not None: 
                        zeros = torch.sum((m.weight == 0).int()).item()
                        size = np.prod(m.weight.shape)
                        self.log['sparse_levels'].append((1- zeros/size))

class SET_zero(SparseModel):


    def find_first_pos(self, tensor, value):
        with torch.no_grad():
            idx = torch.argmin(torch.abs(tensor-value)).item()
        return idx

    def find_last_pos(self, tensor, value):
        """ 
        Invert tensor reference [1]: 
        [1] https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382

        """
        with torch.no_grad():
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
                    new_mask, prune_mask = self.prune(m.weight, self.num_params[idx])
                    with torch.no_grad():
                        self.masks[idx] = new_mask.float()
                        m.weight = torch.nn.Parameter(m.weight * prune_mask.float())
                        # keep track of mean and std of weights
                        self.log['layer_' + str(idx) + '_mean'] = torch.mean(m.weight).item()
                        self.log['layer_' + str(idx) + '_std'] = torch.std(m.weight).item()

        # keep track of mask sizes when debugging
        if self.debug_sparse:
            with torch.no_grad():    
                self.log['mask_sizes'] = [torch.sum(m).item() for m in self.masks]
