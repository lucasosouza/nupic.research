from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ray import tune
import models
import networks

import os
import torch
# torch.manual_seed(32) # run diverse samples


class Dataset():


    def __init__(self,  config=None):

        defaults = dict(
            dataset_name=None, 
            data_dir=None, 
            batch_size_train=128, 
            batch_size_test=128, 
            stats_mean=None, 
            stats_std=None,
        )
        defaults.update(config)
        self.__dict__.update(defaults)

        # expand ~ 
        self.data_dir = os.path.expanduser(self.data_dir)

        # recover mean and std to normalize dataset
        if not self.stats_mean or not self.stats_std:
            tempset = getattr(datasets, self.dataset_name)(root=self.data_dir, 
                                                           train=True,
                                                           transform=transforms.ToTensor(), 
                                                           download=True)
            self.stats_mean = (tempset.data.float().mean().item()/255, )
            self.stats_std = (tempset.data.float().std().item()/255, )
            del tempset

        # set up transformations
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.stats_mean, self.stats_std),
            ]
        )

        # load train set
        train_set = getattr(datasets, self.dataset_name)(root=self.data_dir, 
                                                         train=True,
                                                         transform=transform, 
                                                         download=True)
        self.train_loader = DataLoader(dataset=train_set, 
                                       batch_size=self.batch_size_train, 
                                       shuffle=True)
    
        # load test set
        test_set = getattr(datasets, self.dataset_name)(root=self.data_dir, 
                                                        train=False,
                                                        transform=transform, 
                                                        download=True)
        self.test_loader = DataLoader(dataset=test_set, 
                                      batch_size=self.batch_size_test, 
                                      shuffle=False)

class Trainable(tune.Trainable):
    """
    ray.tune trainable generic class
    Adaptable to any pytorch module
    """

    def __init__(self, config=None, logger_creator=None):
        tune.Trainable.__init__(self, config=config, logger_creator=logger_creator)

    def _setup(self, config):
        network = getattr(networks, config['network'])(config=config)
        self.model = getattr(models, config['model'])(network, config=config)
        self.dataset = Dataset(config=config)
        self.model.setup()

    def _train(self):
        log = self.model.run_epoch(self.dataset)
        return log

    def _save(self, checkpoint_dir):
        self.model.save(checkpoint_dir)

    def _restore(self, checkpoint):
        self.model.restore(checkpoint)








