# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

import os
import itertools
import random
import re
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


CLASSES = (
    "unknown, silence, zero, one, two, three, four, five, six, seven,"
    " eight, nine".split(", ")
)


class VaryingDataLoader(object):
    """Accepts different batch sizes"""

    def __init__(self, dataset, batch_sizes=(1, ), *args, **kwargs):

        if not isinstance(batch_sizes, Iterable):
            batch_sizes = tuple([batch_sizes])
        self.data_loaders = [
            DataLoader(dataset, batch_size, *args, **kwargs)
            for batch_size in batch_sizes
        ]
        self.epoch = 0

    @property
    def data_loader(self):
        return self.data_loaders[self.epoch]

    def __iter__(self):

        data = self.data_loaders[self.epoch]
        self.epoch = min(self.epoch + 1, len(self.data_loaders) - 1)
        return data.__iter__()

    def __len__(self):
        return len(self.data_loader)

    def __getattr__(self, name):

        attrs = self.__dict__.keys()
        if 'data_loaders' in attrs and 'epoch' in attrs:
            if hasattr(self.data_loader, name):
                return getattr(self.data_loader, name)

        # Default
        super().__getattribute__(name)

    def __setattr__(self, name, value):

        attrs = self.__dict__.keys()
        if 'data_loaders' in attrs and 'epoch' in attrs:
            if hasattr(self.data_loader, name):
                self.data_loader.__setattr__(name, value)

        super().__setattr__(name, value)


class PreprocessedSpeechDataset(Dataset):
    """Google Speech Commands dataset preprocessed with with all transforms
    already applied.
    Use the 'process_dataset.py' script to create preprocessed dataset
    """

    def __init__(
            self, root, subset, random_seed=0, classes=CLASSES):
        """
        :param root: Dataset root directory
        :param subset: Which dataset subset to use ("train", "test", "valid", "noise")
        :param classes: List of classes to load. See CLASSES for valid options
        """
        self.classes = classes

        self._root = root
        self._subset = subset

        self.data = None

        # Circular list of all seeds in this dataset
        random.seed(random_seed)
        seeds = [re.search(r'gsc_' + subset + r'(\d+)', e) for e in os.listdir(root)]
        seeds = [int(e.group(1)) for e in seeds if e is not None]
        seeds = seeds if len(seeds) > 0 else [""]
        self._all_seeds = itertools.cycle(seeds if len(seeds) > 0 else "")
        self.num_seeds = len(seeds)

        # Load first seed.
        self.next_seed()

    def __len__(self):
        return len(self.data)

    def __getattr__(self, name):
        return super().__getattr__(name)

    def __iter__(self, name):
        return super().__iter__(name)

    def __getitem__(self, index):
        """Get item from dataset.
        :param index: index in the dataset
        :return: (audio, target) where target is index of the target class.
        :rtype: tuple[dict, int]
        """
        return self.data[index]

    def next_seed(self):
        """Load next seed from disk."""

        seed = str(next(self._all_seeds))
        seed = '0' + seed \
            if (len(seed) == 1) and self._subset == 'test_noise' else seed

        data_path = os.path.join(self._root, 'gsc_' + self._subset + seed + '.npz')

        x, y = np.load(data_path).values()

        x = map(torch.tensor, x)
        y = map(torch.tensor, y)
        self.data = list(zip(x, y))

        return seed


class PreprocessedSpeechDataLoader(VaryingDataLoader):

    def __init__(
        self, root, subset, random_seed=0, classes=CLASSES, batch_sizes=1,
        *args, **kwargs
    ):

        self.dataset = PreprocessedSpeechDataset(
            root, subset, random_seed, classes)

        super().__init__(self.dataset, batch_sizes, *args, *kwargs)

    def __iter__(self):
        iteration = super().__iter__()
        self.dataset.next_seed()
        return iteration

