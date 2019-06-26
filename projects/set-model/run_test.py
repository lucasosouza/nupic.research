from utils import Trainable
import ray
import ray.tune as tune
import os

# alternative initialization based on configuration
config = dict(
    network='MLP',
    # network params
    # hidden_sizes=tune.grid_search([[300, 300, 300], [1000,1000,1000]]),
    hidden_sizes=[300,300,300],
    batch_norm=False, 
    #dropout=tune.grid_search([0, 0.3]), 
    dropout=False,
    bias=False, 
    init_weights=True, 
    num_classes=10,
    input_size=784, 
    # model params
    # model=tune.grid_search(['BaseModel', 'SparseModel']),
    model='SET_faster',
    # model='BaseModel',
    debug_sparse=True,
    dataset_name='MNIST',
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
    resources_per_trial={"cpu": 1, "gpu":0}
)

""""
ongoing notes

"""