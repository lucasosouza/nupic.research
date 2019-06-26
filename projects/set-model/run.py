from utils import Trainable
import ray
import ray.tune as tune
import os

# alternative initialization based on configuration
config = dict(
    network='MLP',
    num_classes=10,
    hidden_sizes=[4000,1000,4000],
    batch_norm=False,
    dropout=0.3,
    bias=True,
    init_weights=True,
    # model=tune.grid_search(['BaseModel', 'SparseModel', 'SET_zero', 'SET_sameDist']),
    model=tune.grid_search(['SET_zero', 'SET_faster']),
    # model='SET_zero',
    dataset_name='CIFAR10',
    input_size=3072, # 784, 
    stats_mean=(0.4914, 0.4822, 0.4465),
    stats_std=(0.2023, 0.1994, 0.2010),
    data_dir='~/nta/datasets',
    device='cuda', # 'cpu',
    optim_alg='SGD',
    debug_sparse=True,
)

# run
ray.init()
tune.run(
    Trainable,
    name='SET_optimization',
    num_samples=1,
    local_dir=os.path.expanduser('~/nta/results'),
    config=config,
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 1000},
    resources_per_trial={"cpu": 1, "gpu": 0.50}
)

""""
ongoing notes

"""