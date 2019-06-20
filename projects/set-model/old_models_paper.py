import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):

    def __init__(self, input_size=784, num_classes=10, init_weights=True,
                 hidden_sizes=[1000, 1000, 1000], device=None):

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
                 hidden_sizes=[1000, 1000, 1000], device=None, bias=False):

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
                 hidden_sizes=[1000, 1000, 1000], device=None, bias=False,
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
        for i, layer in enumerate(linear_layers, 1):
            shape = layer.weight.shape
            sparsity = (shape[0] + shape[1]) / (shape[0] * shape[1])
            # at last layer, can't apply epsilon =20 - see calculations
            if i != len(linear_layers):
                sparsity *= epsilon
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
                    tensor = (torch.randn(m.weight.shape) * 1e-2).to(self.device)
                    tensor = tensor * next(masks)
                m.weight = torch.nn.Parameter(tensor)

    def print_sparse_levels(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                zeros = torch.sum((m.weight == 0).int()).item()
                size = np.prod(m.weight.shape)
                print(1 - zeros / size)


class MLP_set(nn.Module):
    """ Added:
        - SEP Pruning
    """

    def __init__(self, input_size=784, num_classes=10, init_weights=True,
                 hidden_sizes=[1000, 1000, 1000], device=None, bias=False,
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
        for i, layer in enumerate(linear_layers, 1):
            shape = layer.weight.shape
            sparsity = (shape[0] + shape[1]) / (shape[0] * shape[1])
            # at last layer, can't apply epsilon =20 - see calculations
            if i != len(linear_layers):
                sparsity *= epsilon
            mask = torch.rand(shape) < sparsity
            self.masks.append(mask.float().to(self.device))

        if init_weights:
            self._initialize_weights()

    def non_zero(self, tensor):
        """ Support function.
            Accepts a tensor, returns a tensor with only non-zero values
        """

        return tensor.view(-1)[tensor.view(-1).nonzero()].view(-1)

    def forward(self, x):
        return self.classifier(x.view(-1, self.input_size).to(self.device))

    def _initialize_weights(self):
        masks = iter(self.masks)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                with torch.no_grad():
                    tensor = (torch.randn(m.weight.shape) * 1e-2).to(self.device)
                    tensor = tensor * next(masks)
                m.weight = torch.nn.Parameter(tensor)

    def prune(self, A, M, zeta=0.3):
        """ Calculate new weights based on SET approach

            TODO: go over the math again. added gamma to control sparsity
            Last layer is not keeping sparsity level stable, why?

            TODO: can also get from the batch norm layer the running mean and std?

            TODO: evaluate weights over time. are the values stable, increasing or decreasing?
            First evaluation shows it is increasing fast - will need to control for it.

            An option is to ensure new distribution of weights equals the one prior to pruning
            But in this case, the added weights will be likely to be removed in the next pruning

            Other option is to normalize weights for entire layer after pruning. See weight_norm
        """

        with torch.no_grad():
            shape = A.shape
            n_total = np.prod(shape)
            n_zeros = torch.sum(A == 0).item()
            n_values = n_total - n_zeros
            on_perc = n_values / n_total

            # extract mean and std of current weights
            A_nonzero = self.non_zero(A)
            mean = torch.mean(A_nonzero)
            std = torch.std(A_nonzero)

            # calculate threshold and decay connections
            A_abs = torch.abs(A)
            threshold, _ = torch.kthvalue(self.non_zero(A_abs), int(zeta*(n_values)))
            N = A_abs > threshold.item()
            A_prime = A * N.float()

            # initialize matrix of random values 
            R = torch.randn(shape).to(self.device) * std + mean
            # randomly select new connections, zero out conns which had previous weights
            gamma = 0.99
            p_update = zeta * on_perc * gamma / (1 - on_perc)
            P = torch.rand(shape).to(self.device) < p_update
            R_prime = R * P.float() * (M == 0).float()

        # return original masked weights + new weights added
        return (A_prime + R_prime).to(self.device)

    def reinitialize_weights(self):
        """ Reinitialize weights """

        masks = enumerate(self.masks, 0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                mask_idx, mask = next(masks)
                m.weight = torch.nn.Parameter(self.prune(m.weight, mask))
                # print(torch.mean(m.weight))
                self.masks[mask_idx] = (m.weight > 0).float().to(self.device)

    def print_sparse_levels(self):
        print("------------------------")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                zeros = torch.sum((m.weight == 0).int()).item()
                size = np.prod(m.weight.shape)
                print(1 - zeros / size)
        print("------------------------")


class MLP_set_loop(MLP_set):
    """ Modified:
        - Modified pruning from vectorized to loop to compare run time
          Results: loop version is about 15x slower

        TODO: sparsity level is reducing over time. Need to debug prune.
    """

    def __init__(self, input_size=784, num_classes=10, init_weights=True,
                 hidden_sizes=[1000, 1000, 1000], device=None, bias=False,
                 epsilon=20):

        super(MLP_set_loop, self).__init__(input_size=input_size,
                                           num_classes=num_classes,
                                           init_weights=init_weights,
                                           hidden_sizes=hidden_sizes,
                                           device=device,
                                           bias=bias,
                                           epsilon=epsilon)

    def prune(self, A, M, zeta=0.3):
        """ Calculate new weights based on SET approach """

        with torch.no_grad():
            shape = A.shape
            n_total = np.prod(shape)
            n_zeros = torch.sum(A == 0).item()
            n_values = n_total - n_zeros
            perc_on = n_values / n_total

            # extract mean and std of current weights
            # can also get from the batch norm layer the running mean and std?
            A_nonzero = self.non_zero(A)
            mean = torch.mean(A_nonzero)
            std = torch.std(A_nonzero)

            # get threshold
            A_abs = torch.abs(A)
            threshold, _ = torch.kthvalue(self.non_zero(A_abs), int(zeta * (n_values)))

            # calculate weight changes
            prob_on = zeta * perc_on / (1 - perc_on)
            A_prime = torch.zeros(shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    # if 0, there is a prob it will be started
                    if A[i, j] == 0:
                        if torch.rand(1).item() < prob_on:
                            A_prime[i, j] = (torch.randn(1) * std + mean).item()
                    # lesser than threshold are zeroed
                    elif A[i, j] < threshold:
                        A_prime[i, j] = 0
                    # remaining stay the same
                    else:
                        A_prime[i, j] = A[i, j]

        return A_prime.to(self.device)
