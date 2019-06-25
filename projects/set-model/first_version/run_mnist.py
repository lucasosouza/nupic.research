import torch
import torch.optim as optim

torch.manual_seed(32)
device = torch.device('cpu')
if torch.cuda.is_available(): device = torch.device('cuda')

from utils import *
from main import *
from models import *

import logging
logging.basicConfig(filename='mnist.log', level=logging.INFO)
print = logging.info

train_loader, test_loader = load_data('MNIST')
print("Data loaded")

## 1. Run regular MLP on MNIST

model = MLP(device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_func = nn.CrossEntropyLoss()

run_experiment(model, optimizer, loss_func, train_loader, test_loader,
               device=device)

## 2. Replicate dense model

# switch to SGD with weight decay
model = MLP_dense(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=2e-4, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

run_experiment(model, optimizer, loss_func, train_loader, test_loader,
               device=device)

## 3. Implement fixed sparsity

# switch to SGD with weight decay
model = MLP_sparse(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=2e-4, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

run_experiment(model, optimizer, loss_func, train_loader, test_loader,
               device=device, sparse=True)

model.print_sparse_levels()

## 4. Implement SET

# switch to SGD with weight decay
model = MLP_set(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=2e-4, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

run_experiment(model, optimizer, loss_func, train_loader, test_loader,
               device=device, sparse=True, sep=True)

model.print_sparse_levels()
