from utils import Trainable
import ray
import ray.tune as tune
import os

# alternative initialization based on configuration
config = dict(
    network='resnet18',
    # network params
    # hidden_sizes=tune.grid_search([[300, 300, 300], [1000,1000,1000]]),
    # hidden_sizes=[300,300,300],
    # batch_norm=False, 
    #dropout=tune.grid_search([0, 0.3]), 
    # dropout=False,
    # bias=False, 
    # init_weights=True, 
    num_classes=10,
    # input_size=784, 
    # model params
    # model=tune.grid_search(['BaseModel', 'SparseModel']),
    model=tune.grid_search('[BaseModel, SparseModel, SET_faster]'),
    # model='BaseModel',
    debug_sparse=True,
    dataset_name='CIFAR10',
    # input_size=(3,32,32), # 784, 
    stats_mean=(0.4914, 0.4822, 0.4465),
    stats_std=(0.2023, 0.1994, 0.2010),    
    data_dir='~/nta/datasets',
    device='cpu',
    optim_alg='SGD',
)

# run
ray.init()
tune.run(
    Trainable,
    name='SET_local_test',
    num_samples=1,
    local_dir=os.path.expanduser('~/nta/results'),
    config=config,
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 100},
    resources_per_trial={"cpu": 1, "gpu":0.3}
)

""""
ongoing notes

"""