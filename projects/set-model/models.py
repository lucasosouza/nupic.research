import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedulers
import numpy as np

class BaseModel():

    def __init__(self, network, config={}):

        defaults = dict(
            optim_alg='SGD',
            learning_rate=0.1,
            momentum=0.9, 
            device='cpu',
            lr_scheduler=False,
            debug_sparse=False,
            debug_weights=False,
            start_sparse=None,
            end_sparse=None,
        )
        defaults.update(config)
        self.__dict__.update(defaults)

        # init remaining
        self.device = torch.device(self.device)
        self.network = network.to(self.device)

    def setup(self):      

        # init optimizer
        if self.optim_alg == 'Adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        elif self.optim_alg == 'SGD':
            self.optimizer = optim.SGD(self.network.parameters(), 
                lr=self.learning_rate, momentum=self.momentum)

        # add a learning rate scheduler
        if self.lr_scheduler:
            self.lr_scheduler = schedulers.MultiStepLR(self.optimizer, 
                milestones=self.lr_milestones, gamma=self.lr_gamma)

        # init loss function
        self.loss_func = nn.CrossEntropyLoss()

    def run_epoch(self, dataset):
        self.log={}
        self.network.train()
        self._run_one_pass(dataset.train_loader, train=True)
        self.network.eval()
        self._run_one_pass(dataset.test_loader, train=False)
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return self.log

    def _run_one_pass(self, loader, train=True):
        epoch_loss = 0
        correct = 0
        for inputs, targets in loader:
            # setup for training
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            # training loop
            with torch.set_grad_enabled(train):
                # forward + backward + optimize
                outputs = self.network(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(targets == preds).item()
                loss = self.loss_func(outputs, targets)
                if train:
                    loss.backward()
                    self.optimizer.step()

            # keep track of loss
            epoch_loss += loss.item() * inputs.size(0)

        # store loss and acc at each pass
        loss = epoch_loss / len(loader.dataset)
        acc = correct / len(loader.dataset)
        if train:
            self.log['train_loss'] = loss
            self.log['train_acc'] = acc
        else:
            self.log['val_loss'] = loss
            self.log['val_acc'] = acc

        if train and self.debug_weights:
            for idx, m in enumerate(self.network.modules()):
                if self.has_params(m):
                    # keep track of mean and std of weights
                    self.log['layer_' + str(idx) + '_mean'] = torch.mean(m.weight).item()
                    self.log['layer_' + str(idx) + '_std'] = torch.std(m.weight).item()

    @staticmethod
    def has_params(module):
        return isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)

    def save(self):
        pass

    def restore(self):
        pass

class SparseModel(BaseModel):

    def setup(self):
        super(SparseModel, self).setup()

        # add specific defaults
        if 'epsilon' not in self.__dict__:
            self.epsilon = 20

        with torch.no_grad():
            # calculate sparsity masks
            self.masks = []
            self.denseness = []
            self.num_params = [] # added for paper implementation

            # define sparse modules
            self.sparse_modules = []
            for m in list(self.network.modules())[self.start_sparse:self.end_sparse]:
                if self.has_params(m):
                    self.sparse_modules.append(m)

            # initialize masks
            for m in self.sparse_modules:
                shape = m.weight.shape
                on_perc = self.epsilon * np.sum(shape)/np.prod(shape)
                mask = (torch.rand(shape) < on_perc).float().to(self.device)
                m.weight.data *= mask
                self.masks.append(mask)
                self.denseness.append(on_perc)
                self.num_params.append(torch.sum(mask).item())  

    def _run_one_pass(self, loader, train):
        """ TODO: reimplement by calling super and passing a hook """ 

        epoch_loss = 0
        epoch_correct = 0
        for inputs, targets in loader:
            # setup for training
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()

            # forward + backward + optimize
            with torch.set_grad_enabled(train):
                outputs = self.network(inputs)
                _, preds = torch.max(outputs, 1)
                epoch_correct += torch.sum(targets == preds).item()
                loss = self.loss_func(outputs, targets)
                if train:
                    loss.backward()
                    # zero the gradients for dead connections
                    masks = iter(self.masks)
                    for m in self.sparse_modules:
                        mask = next(masks, None)
                        m.weight.grad *= mask
                    self.optimizer.step()

            # keep track of loss and accuracy
            epoch_loss += loss.item() * inputs.size(0)

        # store loss and acc at each pass
        loss = epoch_loss / len(loader.dataset)
        acc = epoch_correct / len(loader.dataset)
        if train:
            self.log['train_loss'] = loss
            self.log['train_acc'] = acc
        else:
            self.log['val_loss'] = loss
            self.log['val_acc'] = acc

        # monitor weights (TODO: move to callbacks)
        if train and self.debug_weights:
            for idx, m in enumerate(self.network.modules()):
                if self.has_params(m):
                    # keep track of mean and std of weights
                    self.log['layer_' + str(idx) + '_mean'] = torch.mean(m.weight).item()
                    self.log['layer_' + str(idx) + '_std'] = torch.std(m.weight).item()

        # add monitoring of sparse levels
        if train and self.debug_sparse: 
            self._log_sparse_levels()


    def _log_sparse_levels(self):
        layers = iter(range(len(self.masks)))
        with torch.no_grad():
            for m in self.sparse_modules:
                idx = next(layers)
                zeros = torch.sum((m.weight == 0).int()).item()
                size = np.prod(m.weight.shape)
                self.log['sparse_levels_l' + str(idx+1)] = 1- zeros/size

class SET_faster(SparseModel):

    def _run_one_pass(self, loader, train):
        super(SET_faster, self)._run_one_pass(loader, train)
        if train:
            self.reinitialize_weights()

    def prune(self, A, num_params, zeta=0.3):
        """ Calculate new weights based on SET approach 
            A vectorized version aimed at keeping the mask with the similar level of sparsity 

        """
        with torch.no_grad():

            # NOTE: another approach is counting how many numbers to prune (see old model)
            # calculate thresholds and decay weak connections
            A_pos = A[A>0]
            pos_threshold, _ = torch.kthvalue(A_pos, int(zeta*len(A_pos)))
            A_neg = A[A<0]
            neg_threshold, _ = torch.kthvalue(A_neg, int((1-zeta)*len(A_neg)))
            prune_mask = ((A >= pos_threshold) | (A <= neg_threshold)).to(self.device)

            # change mask to add new weights
            num_add = num_params - torch.sum(prune_mask).item()
            current_sparsity = torch.sum(A==0).item()
            p_add = num_add / max(current_sparsity, num_add) # avoid div by zero
            P = torch.rand(A.shape).to(self.device) < p_add        
            new_mask = prune_mask | (P & (A==0))

        return new_mask, prune_mask

    def reinitialize_weights(self):        
        """ Reinitialize weights """

        layers = iter(range(len(self.masks)))
        for m in self.sparse_modules:
            idx = next(layers)
            layer_weights = m.weight.clone().detach()                
            new_mask, prune_mask = self.prune(layer_weights, self.num_params[idx])
            with torch.no_grad():
                self.masks[idx] = new_mask.float()
                m.weight.data *= prune_mask.float()

        # keep track of mask sizes when debugging
        if self.debug_sparse:
            for idx, m in enumerate(self.masks):    
                self.log['mask_sizes_l' + str(idx+1)] = torch.sum(m).item()





