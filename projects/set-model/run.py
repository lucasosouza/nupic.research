from utils import Trainable
import ray
import ray.tune as tune
import os

# alternative initialization based on configuration
config = dict(
    network='MLP',
    input_size=784,
    num_classes=10,
    hidden_sizes=[1000,1000,1000],
    batch_norm=True,
    dropout=True,
    bias=False,
    init_weights=True,
    # model=tune.grid_search(['BaseModel', 'SparseModel', 'SET_zero', 'SET_sameDist']),
    model=tune.grid_search(['BaseModel', 'SparseModel', 'SET_zero']),
    dataset_name='MNIST',
    data_dir='~/nta/datasets',
    device='cpu',
    optim_alg='SGD',
    debug_sparse=True,
)

# run
ray.init()
tune.run(
    Trainable,
    name='SET_local_test',
    num_samples=2,
    local_dir=os.path.expanduser('~/nta/results'),
    config=config,
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 500},
)

""""
ongoing notes

"""