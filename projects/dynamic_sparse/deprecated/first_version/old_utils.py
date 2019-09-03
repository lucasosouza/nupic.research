from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np

data_dir = os.path.expanduser("~/nta/results")
  

def load_data(dataset, batch_size_train=128, batch_size_test=128):
    """ Load a torchvision standard dataset """
    
    # get mean and std
    tempset = getattr(datasets, dataset)(root=data_dir, train=True,
                                         transform=transforms.ToTensor(), download=True)
    datastats_mean = tempset.train_data.float().mean().item()/255
    datastats_std = tempset.train_data.float().std().item()/255

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((datastats_mean,), (datastats_std,)),
        ]
    )

    train_set = getattr(datasets, dataset)(root=data_dir, train=True,
                                           transform=transform, 
                                           download=True)
    
    test_set = getattr(datasets, dataset)(root=data_dir, train=False,
                                          transform=transform, 
                                          download=True)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size_train,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size_test, 
                             shuffle=False)
    
    return train_loader, test_loader



