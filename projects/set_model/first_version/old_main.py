import torch
import torch.nn as nn

# import logging
# print = logging.info

def run_epoch(model, loss_func, loader, optimizer=None, train=True, 
              sparse=False, sep=False, device=None):
    """ Run one epoch"""

    epoch_loss = 0
    num_samples = 0
    correct = 0
    for data in loader:
        # get inputs and label
        inputs, targets = data
        targets = targets.to(device)
        
        # zero gradients
        if train: optimizer.zero_grad()
                
        # forward + backward + optimize
        outputs = model(inputs)
        correct += (targets==torch.max(outputs, dim=1)[1]).sum().item()
        loss = loss_func(outputs, targets)
        if train:
            loss.backward()
            if sparse:
                # zero the gradients for dead connections
                masks = iter(model.masks)
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        mask = next(masks, None)
                        if mask is not None:
                            m.weight.grad *= mask
            optimizer.step()

        # keep track of loss
        epoch_loss += loss.item()
        num_samples += inputs.shape[0]
    
    if sep:
        # at the end of epoch, reinitialize weights
        model.reinitialize_weights()        
        
    return (epoch_loss / num_samples) * 1000, correct/num_samples

def run_experiment(model, optimizer, loss_func,
                   train_loader, test_loader, device,
                   epochs=250, sparse=False, sep=False,
                   print_sparse_levels=False):
    """ Run one experiment """

    max_train_acc = 0
    max_val_acc = 0
    for epoch in range(epochs):    
        model.train()
        train_loss, train_acc = run_epoch(model, loss_func, train_loader, 
                                          optimizer, train=True, device=device,
                                          sparse=sparse, sep=sep)
        max_train_acc = max(train_acc, max_train_acc)
        
        model.eval()
        val_loss, val_acc     = run_epoch(model, loss_func, test_loader, 
                                          device=device, train=False)
        max_val_acc = max(val_acc, max_val_acc)
        
        print('Epoch: %d Train loss: %.4f Train Acc: %.4f Val loss: %.4f Val Acc: %.4f' % 
              (epoch + 1, train_loss, train_acc, val_loss, val_acc))
        if print_sparse_levels: model.print_sparse_levels()
        
    print('\nFinal - Best Train Acc: %.4f Best Val acc: %.4f' % (max_train_acc, max_val_acc))
    
