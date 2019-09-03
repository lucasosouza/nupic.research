# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

from collections.abc import Iterable
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedulers
from .base_models import *

# TO FIX
import sys
sys.path.append("..")
from layers import calc_sparsity, DSConv2d, SparseConv2d

class DSNNMixedHeb(DSNNHeb):
    """Improved results compared to DSNNHeb"""

    def prune(self, weight, grad, num_params, corr, idx=0):
        """
        Grow by correlation
        Prune by magnitude
        """
        with torch.no_grad():

            # print("corr dimension", corr.shape)
            # print("weight dimension", weight.shape)

            # transpose to fit the weights
            corr = corr.t()

            tau = self.hebbian_prune_perc
            # decide which weights to remove based on correlation
            kth = int((1 - tau) * np.prod(corr.shape))
            corr_threshold, _ = torch.kthvalue(corr.contiguous().view(-1), kth)
            hebbian_keep_mask = (corr > corr_threshold).to(self.device)

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

            # no gradient mask, just a keep mask
            keep_mask = weight_keep_mask & hebbian_keep_mask
            self.log["weight_keep_mask_l" + str(idx)] = torch.sum(keep_mask).item()

            # calculate number of parameters to add
            num_add = max(
                num_params - torch.sum(keep_mask).item(), 0
            )  # TODO: debug why < 0
            self.log["missing_weights_l" + str(idx)] = num_add
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

class DSNNConvHeb(DSNNMixedHeb):
    """
    Similar to other sparse models, but the focus here is on convolutional layers as
    opposed to dense layers.
    """

    log_attrs = [
        'pruning_iterations',
        'kept_frac',
        'prune_mask_sparsity',
        'keep_mask_sparsity',
        'weight_sparsity',
        'last_coactivations',
    ]

    def is_sparse(self, module):
        if isinstance(module, DSConv2d):
            return "sparse_conv"

    def setup(self):
        super().setup()
        # find sparse layers
        self.sparse_conv_modules = []
        for m in list(self.network.modules()):
            if self.is_sparse(m):
                self.sparse_conv_modules.append(m)
        # print(self.sparse_conv_modules)

    def _post_epoch_updates(self, dataset=None):
        """Only change in the model is here. 
        In order to work, need to use networks which have DSConv2d layer
        which network is being used?"""

        print("calling post epoch")
        super()._post_epoch_updates(dataset)

        # go through named modules
        for idx, module in enumerate(self.sparse_conv_modules):
            # if it is a dsconv layer
            # print("layer type: ", module.__class__)
            # print(isinstance(module, DSConv2d))
            # print("progressing connections")
            # Log coactivation before pruning - otherwise they get reset.
            self.log['hist_' + 'coactivations_' + str(idx)] = module.coactivations
            # Prune. Then log some params.
            module.progress_connections()
            print("progressing")
            for attr in self.log_attrs:
                value = getattr(module, attr) if hasattr(module, attr) else -2
                if isinstance(value, Iterable):
                    attr = 'hist_' + attr
                self.log[attr + '_' + str(idx)] = value

            if isinstance(module, (DSConv2d, SparseConv2d)):
                self.log['sparsity_' + str(idx)] = calc_sparsity(module.weight)


class DSNNConvOnlyHeb(BaseModel):
    """
    Similar to other sparse models, but the focus here is on convolutional layers as
    opposed to dense layers.
    """

    log_attrs = [
        'pruning_iterations',
        'kept_frac',
        'prune_mask_sparsity',
        'keep_mask_sparsity',
        'weight_sparsity',
        'last_coactivations',
    ]

    def is_sparse(self, module):
        if isinstance(module, DSConv2d):
            return "sparse_conv"

    def setup(self):
        super().setup()
        # find sparse layers
        self.sparse_conv_modules = []
        for m in list(self.network.modules()):
            if self.is_sparse(m):
                self.sparse_conv_modules.append(m)
        # print(self.sparse_conv_modules)

    def _post_epoch_updates(self, dataset=None):
        """Only change in the model is here. 
        In order to work, need to use networks which have DSConv2d layer
        which network is being used?"""

        print("calling post epoch")
        super()._post_epoch_updates(dataset)

        # go through named modules
        for idx, module in enumerate(self.sparse_conv_modules):
            # if it is a dsconv layer
            # print("layer type: ", module.__class__)
            # print(isinstance(module, DSConv2d))
            # print("progressing connections")
            # Log coactivation before pruning - otherwise they get reset.
            self.log['hist_' + 'coactivations_' + str(idx)] = module.coactivations
            # Prune. Then log some params.
            module.progress_connections()
            print("progressing")
            for attr in self.log_attrs:
                value = getattr(module, attr) if hasattr(module, attr) else -2
                if isinstance(value, Iterable):
                    attr = 'hist_' + attr
                self.log[attr + '_' + str(idx)] = value

            if isinstance(module, (DSConv2d, SparseConv2d)):
                self.log['sparsity_' + str(idx)] = calc_sparsity(module.weight)

