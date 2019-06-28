from utils import Trainable, download_dataset
import ray
import ray.tune as tune
import os

# alternative initialization based on configuration
config = dict(
    network='vgg19_bn',
    num_classes=10,
    model=tune.grid_search(['BaseModel', 'SparseModel', 'SET_faster']),
    epsilon=50,
    start_sparse=1,
    momentum=0.9,
    learning_rate=1e-2,
    lr_scheduler='MultiStepLR',
    lr_milestones=[81,122],
    lr_gamma=0.10,
    dataset_name='CIFAR10',
    augment_images=True,
    stats_mean=(0.4914, 0.4822, 0.4465),
    stats_std=(0.2023, 0.1994, 0.2010),
    data_dir='~/nta/datasets',
    device='cuda', # 'cpu',
    optim_alg='SGD',
    debug_weights=True,
    debug_sparse=True,
)

# run
download_dataset(config)
ray.init()
tune.run(
    Trainable,
    name='SET_optimization',
    num_samples=1,
    local_dir=os.path.expanduser('~/nta/results'),
    config=config,
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 400},
    resources_per_trial={"cpu": 1, "gpu": 0.33}
)


