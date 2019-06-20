import torch
import torch.optim as optim

torch.manual_seed(32)
device = torch.device('cpu')
if torch.cuda.is_available(): device = torch.device('cuda')

from utils import *
from main import *
from models import *

import logging
logging.basicConfig(filename='compare.logfile', level=logging.INFO)
print = logging.info

train_loader, test_loader = load_data('FashionMNIST')
print("Data loaded")

epochs = 100

# regular model
model = MLP_set(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=2e-4, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

run_experiment(model, optimizer, loss_func, train_loader, test_loader,
               device=device, sparse=True, sep=True, epochs=epochs)

model.print_sparse_levels()

# reinit weights to zero
model = MLP_set_zero(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=2e-4, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

run_experiment(model, optimizer, loss_func, train_loader, test_loader,
               device=device, sparse=True, sep=True, epochs=epochs)

model.print_sparse_levels()

# reinit weights to the same initial distribution
model = MLP_set_samedist(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=2e-4, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

run_experiment(model, optimizer, loss_func, train_loader, test_loader,
               device=device, sparse=True, sep=True, epochs=epochs)

model.print_sparse_levels()

