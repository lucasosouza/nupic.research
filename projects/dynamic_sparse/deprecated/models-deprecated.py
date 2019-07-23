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



class DSNN(SparseModel):
    """
    Dynamically sparse neural networks.
    Our improved version of SET
    """

    def setup(self):
        super(DSNN, self).setup()
        self.added_synapses = [None for m in self.masks]
        self.last_gradients = [None for m in self.masks]

        # initializae sign to 1 if to be flipped later
        self.prune_grad_sign = -1
        if self.flip:
            self.prune_grad_sign = 1
            self.flip_epoch = 30

    def _post_epoch_updates(self, dataset=None):
        super(DSNN, self)._post_epoch_updates(dataset)

        # flip at a fixed interval
        if self.flip:
            if self.current_epoch == self.flip_epoch and self.prune_grad_sign == 1:
                self.prune_grad_sign = -1

        # update only when learning rate updates
        # froze this for now
        # if self.current_epoch in self.lr_milestones:
        #     # decay pruning interval, inversely proportional with learning rate
        #     self.pruning_interval = max(self.pruning_interval,
        #         int((self.pruning_interval * (1/self.lr_gamma))/3))

    def _run_one_pass(self, loader, train, noise=False):
        """TODO: reimplement by calling super and passing a hook"""
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
                    for idx, (mask, m) in enumerate(
                        zip(self.masks, self.sparse_modules)
                    ):
                        m.weight.grad *= mask
                        # save gradients before any operation
                        # TODO: will need to integrate over several epochs later
                        self.last_gradients[idx] = m.weight.grad

                    self.optimizer.step()

            # keep track of loss and accuracy
            epoch_loss += loss.item() * inputs.size(0)

        # store loss and acc at each pass
        loss = epoch_loss / len(loader.dataset)
        acc = epoch_correct / len(loader.dataset)
        if train:
            self.log["train_loss"] = loss
            self.log["train_acc"] = acc
        else:
            if noise:
                self.log["noise_loss"] = loss
                self.log["noise_acc"] = acc
            else:
                self.log["val_loss"] = loss
                self.log["val_acc"] = acc

        # add monitoring of weights
        if train and self.debug_weights:
            self._log_weights()

        # add monitoring of sparse levels
        if train and self.debug_sparse:
            self._log_sparse_levels()

        # dynamically decide pruning interval
        if train:
            # no dynamic interval at this moment
            # if self.current_epoch % self.pruning_interval == 0:
            self.reinitialize_weights()

    def prune(self, weight, grad, num_params, zeta=0.30, idx=0):
        """
        Calculate new weight based on SET approach weight vectorized version
        aimed at keeping the mask with the similar level of sparsity.

        Changes:
        - higher zeta
        - two masks: one based on weights, another based on gradients
        """
        with torch.no_grad():

            # calculate weight mask
            zeta = self.weight_prune_perc
            weight_pos = weight[weight > 0]
            pos_threshold, _ = torch.kthvalue(
                weight_pos, max(int(zeta * len(weight_pos)), 1)
            )
            weight_neg = weight[weight < 0]
            neg_threshold, _ = torch.kthvalue(
                weight_neg, max(int((1 - zeta) * len(weight_neg)), 1)
            )
            weight_keep_mask = (weight >= pos_threshold) | (weight <= neg_threshold)
            weight_keep_mask.to(self.device)
            self.log["weight_keep_mask_l" + str(idx)] = torch.sum(
                weight_keep_mask
            ).item()

            # calculate gradient mask
            kappa = self.grad_prune_perc
            grad = grad * self.prune_grad_sign * torch.sign(weight)
            deltas = grad.view(-1)
            grad_treshold, _ = torch.kthvalue(deltas, max(int(kappa * len(deltas)), 1))
            grad_keep_mask = (grad >= grad_treshold).to(self.device)
            # keep only those which are in the original weight matrix
            grad_keep_mask = grad_keep_mask & (weight != 0)
            self.log["grad_keep_mask_l" + str(idx)] = torch.sum(grad_keep_mask).item()

            # combine both masks
            keep_mask = weight_keep_mask & grad_keep_mask

            # change mask to add new weight
            num_add = num_params - torch.sum(keep_mask).item()
            self.log["missing_weights_l" + str(idx)] = num_add
            current_sparsity = torch.sum(weight == 0).item()
            self.log["zero_weights_l" + str(idx)] = current_sparsity
            p_add = num_add / max(current_sparsity, 1)  # avoid div by zero
            probs = torch.rand(weight.shape).to(self.device) < p_add
            new_synapses = probs & (weight == 0)
            new_mask = keep_mask | new_synapses
            self.log["added_synapses_l" + str(idx)] = torch.sum(new_synapses).item()

        # track added connections
        return new_mask, keep_mask, new_synapses

    def reinitialize_weights(self):
        """Reinitialize weights."""
        for idx, (m, grad) in enumerate(zip(self.sparse_modules, self.last_gradients)):
            new_mask, keep_mask, new_synapses = self.prune(
                m.weight.clone().detach(), grad, self.num_params[idx], idx=idx
            )
            with torch.no_grad():
                self.masks[idx] = new_mask.float()
                m.weight.data *= keep_mask.float()

                # keep track of added synapes
                if self.debug_sparse:
                    # count new synapses at this round
                    # total_new = torch.sum(new_synapses).item()
                    # total = np.prod(new_synapses.shape)
                    # self.log["added_synapses_l" + str(idx)] = total_new
                    # count how many synapses from last round have survived
                    if self.added_synapses[idx] is not None:
                        total_added = torch.sum(self.added_synapses[idx]).item()
                        surviving = torch.sum(
                            self.added_synapses[idx] & keep_mask
                        ).item()
                        if total_added:
                            self.log["surviving_synapses_l" + str(idx)] = (
                                surviving / total_added
                            )
                # keep track of new synapses to count surviving on next round
                self.added_synapses[idx] = new_synapses

        # keep track of mask sizes when debugging
        if self.debug_sparse:
            for idx, m in enumerate(self.masks):
                self.log["mask_sizes_l" + str(idx)] = torch.sum(m).item()
                


class DSNNHeb(SparseModel):
    """
    DSNNHeb
    Grow: by correlation
    Prune: by magnitude

    Improved results compared to regular SET
    """

    def setup(self):
        super(DSNNHeb, self).setup()
        self.added_synapses = [None for m in self.masks]
        self.last_gradients = [None for m in self.masks]

        # # initializae sign to 1 if to be flipped later
        # self.prune_grad_sign = -1
        # if self.flip:
        #     self.prune_grad_sign = 1
        #     self.flip_epoch = 30

        # add specific defaults
        new_defaults = dict(
            pruning_active=True,
            pruning_es=True,
            pruning_es_patience=0,
            pruning_es_window_size=6,
            pruning_es_threshold=0.02,
            pruning_interval=1,
            hebbian_prune_perc=0,
        )
        new_defaults = {k: v for k, v in new_defaults.items() if k not in self.__dict__}
        self.__dict__.update(new_defaults)

        # initialize hebbian learning
        self.network.hebbian_learning = True
        # count number of cycles, compare with patience
        self.pruning_es_cycles = 0
        self.last_survival_ratios = deque(maxlen=self.pruning_es_window_size)

    def _post_epoch_updates(self, dataset=None):
        super(DSNNHeb, self)._post_epoch_updates(dataset)

        # flip at a fixed interval
        # if self.flip:
        #     if self.current_epoch == self.flip_epoch and self.prune_grad_sign == 1:
        #         self.prune_grad_sign = -1

        # zero out correlations
        self.network.correlations = []

        # update only when learning rate updates
        # froze this for now
        # if self.current_epoch in self.lr_milestones:
        #     # decay pruning interval, inversely proportional with learning rate
        #     self.pruning_interval = max(self.pruning_interval,
        #         int((self.pruning_interval * (1/self.lr_gamma))/3))

    def _run_one_pass(self, loader, train, noise=False):
        """TODO: reimplement by calling super and passing a hook"""
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
                    for idx, (mask, m) in enumerate(
                        zip(self.masks, self.sparse_modules)
                    ):
                        m.weight.grad *= mask
                        # save gradients before any operation
                        # TODO: will need to integrate over several epochs later
                        self.last_gradients[idx] = m.weight.grad

                    self.optimizer.step()

            # keep track of loss and accuracy
            epoch_loss += loss.item() * inputs.size(0)

        # store loss and acc at each pass
        loss = epoch_loss / len(loader.dataset)
        acc = epoch_correct / len(loader.dataset)
        if train:
            self.log["train_loss"] = loss
            self.log["train_acc"] = acc
        else:
            if noise:
                self.log["noise_loss"] = loss
                self.log["noise_acc"] = acc
            else:
                self.log["val_loss"] = loss
                self.log["val_acc"] = acc

        # add monitoring of weights
        if train and self.debug_weights:
            self._log_weights()

        # add monitoring of sparse levels
        if train and self.debug_sparse:
            self._log_sparse_levels()

        # dynamically decide pruning interval
        if train:
            # no dynamic interval at this moment
            # if self.current_epoch % self.pruning_interval == 0:
            self.reinitialize_weights()

    def prune(self, weight, grad, num_params, corr, idx=0):
        """
        Grow by correlation
        Prune by correlation and magnitude

        Error added contiguous to fix
        RuntimeError: invalid argument 2: view size is not compatible
        with input tensor's size and stride (at least one dimension
        spans across two contiguous subspaces). Call .contiguous()
        before .view(). at /pytorch/aten/src/THC/generic/THCTensor.cpp:209
        """
        with torch.no_grad():

            # calculate weight mask
            zeta = self.weight_prune_perc
            weight_pos = weight[weight > 0]
            pos_threshold, _ = torch.kthvalue(
                weight_pos, max(int(zeta * len(weight_pos)), 1)
            )
            weight_neg = weight[weight < 0]
            neg_threshold, _ = torch.kthvalue(
                weight_neg, max(int((1 - zeta) * len(weight_neg)), 1)
            )
            weight_keep_mask = (weight >= pos_threshold) | (weight <= neg_threshold)
            weight_keep_mask.to(self.device)
            self.log["weight_keep_mask_l" + str(idx)] = torch.sum(
                weight_keep_mask
            ).item()

            # no gradient mask, just a keep mask
            keep_mask = weight_keep_mask

            # calculate number of parameters to add
            num_add = num_params - torch.sum(keep_mask).item()
            self.log["missing_weights_l" + str(idx)] = num_add
            # transpose to fit the weights
            corr = corr.t()
            # remove the ones which will already be kept
            corr *= (keep_mask == 0).float()
            # get kth value, based on how many weights to add, and calculate mask
            kth = int(np.prod(corr.shape) - num_add)
            # contiguous()
            corr_threshold, _ = torch.kthvalue(corr.contiguous().view(-1), kth)
            add_mask = (corr > corr_threshold).to(self.device)

            new_mask = keep_mask | add_mask
            self.log["added_synapses_l" + str(idx)] = torch.sum(add_mask).item()

        # track added connections
        return new_mask, keep_mask, add_mask

    def reinitialize_weights(self):
        """Reinitialize weights."""
        # only run if still learning and if at the right interval
        # current epoch is 1-based indexed
        if self.pruning_active and (self.current_epoch % self.pruning_interval) == 0:

            # keep track of added synapes
            survival_ratios = []

            for idx, (m, grad, corr) in enumerate(
                zip(self.sparse_modules, self.last_gradients, self.network.correlations)
            ):
                new_mask, keep_mask, new_synapses = self.prune(
                    m.weight.clone().detach(), grad, self.num_params[idx], corr, idx=idx
                )
                with torch.no_grad():
                    self.masks[idx] = new_mask.float()
                    m.weight.data *= keep_mask.float()

                    # count how many synapses from last round have survived
                    if self.added_synapses[idx] is not None:
                        total_added = torch.sum(self.added_synapses[idx]).item()
                        surviving = torch.sum(
                            self.added_synapses[idx] & keep_mask
                        ).item()
                        if total_added:
                            survival_ratio = surviving / total_added
                            survival_ratios.append(survival_ratio)
                            # log if in debug sparse mode
                            if self.debug_sparse:
                                self.log[
                                    "surviving_synapses_l" + str(idx)
                                ] = survival_ratio

                    # keep track of new synapses to count surviving on next round
                    self.added_synapses[idx] = new_synapses

            # early stop (alternative - keep a moving average)
            # ignore the last layer for now
            mean_survival_ratio = np.mean(
                survival_ratios[:-1]
            )
            if not np.isnan(mean_survival_ratio):
                self.last_survival_ratios.append(mean_survival_ratio)
                if self.debug_sparse:
                    self.log["surviving_synapses_avg"] = mean_survival_ratio
                if self.pruning_es:
                    ma_survival = (
                        np.sum(list(self.last_survival_ratios))
                        / self.pruning_es_window_size
                    )
                    if ma_survival < self.pruning_es_threshold:
                        self.pruning_es_cycles += 1
                        self.last_survival_ratios.clear()
                    if self.pruning_es_cycles > self.pruning_es_patience:
                        self.pruning_active = False

            # keep track of mask sizes when debugging
            if self.debug_sparse:
                for idx, m in enumerate(self.masks):
                    self.log["mask_sizes_l" + str(idx)] = torch.sum(m).item()

