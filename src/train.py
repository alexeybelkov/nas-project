import torch
import numpy as np
from copy import deepcopy
from .utils import weight_init, num_params, set_params
import torch.nn.functional as torchF
from .modules import *
from .tenas import TENAS


def train(model, device, train_loader, val_loader, test_loader,
          learning_rate, num_epochs, launches):

    launch_losses = []
    for launch in range(launches): 

        model_ = deepcopy(model).to(device)
        model_.apply(weight_init)
        optimizer = torch.optim.Adam(model_.parameters(), lr=learning_rate)
        val_losses = []
        test_losses = []
        for epoch in range(num_epochs):
    
            model_.train()
            for batch in train_loader:
                batch = torch.stack(tuple(batch.values()), dim=1).to(device)
                inputs, targets = batch[:, :-1], batch[:, -1]
                
                model.zero_grad()
                preds = model_(inputs).flatten()
                
                loss = torchF.mse_loss(targets, preds)
                loss.backward()
                optimizer.step()

            val_losses_epoch = []
            test_losses_epoch = []
            model_.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = torch.stack(tuple(batch.values()), dim=1).to(device)
                    inputs, targets = batch[:, :-1], batch[:, -1]
                    
                    preds = model_(inputs).flatten()
                    loss = torchF.mse_loss(targets, preds)
                    val_losses_epoch.append(loss.item())
    
                for batch in test_loader:
                    batch = torch.stack(tuple(batch.values()), dim=1).to(device)
                    inputs, targets = batch[:, :-1], batch[:, -1]
                    
                    preds = model_(inputs).flatten()
                    loss = torchF.mse_loss(targets, preds)
                    test_losses_epoch.append(loss.item())

            val_losses.append(np.mean(val_losses_epoch))
            test_losses.append(np.mean(test_losses_epoch))

        final_test_loss = test_losses[np.argmin(val_losses)]
        launch_losses.append(final_test_loss)

        model_.eval()

    return np.mean(launch_losses), np.std(launch_losses)


def train_unpruned_mlp(device, input_dim, hidden_dim,
                       lr, num_epochs, num_launches,
                       num_workers, batch_size,
                       num_threads, seed, num_batch=None, **kwargs):
    """
    Output: Mean MSE on Test, Std MSE on Test, Number of params in model, Architecture 
    """
    mlp = MLP(input_dim, hidden_dim).to(device)

    n_param = num_params(mlp)

    train_loader, val_loader, test_loader = set_params(num_workers, batch_size, num_threads, seed)

    return train(mlp, device, train_loader, val_loader, test_loader,
                 lr, num_epochs, num_launches), n_param, str(mlp)


def train_classic_mlp(device, input_dim, hidden_dim,
                      lr, num_epochs, num_launches,
                      num_workers, batch_size,
                      num_threads, seed, num_batch=None, **kwargs):
    mlp = ClassicMLP(input_dim, hidden_dim).to(device)

    n_param = num_params(mlp)

    train_loader, val_loader, test_loader = set_params(num_workers, batch_size, num_threads, seed)

    return train(mlp, device, train_loader, val_loader, test_loader,
                 lr, num_epochs, num_launches), n_param, str(mlp)


def train_pruned_mlp(device, input_dim, hidden_dim,
                     lr, num_epochs, num_launches,
                     num_workers, batch_size,
                     num_threads, seed, space_size, num_batch=1, **kwargs):
    mlp = MLP(input_dim, hidden_dim).to(device)
    
    train_loader, val_loader, test_loader = set_params(num_workers, batch_size, num_threads, seed)

    pruned_mlp = TENAS(train_loader, mlp, space_size, device, num_batch)

    n_param = num_params(pruned_mlp)

    return train(pruned_mlp, device, train_loader, val_loader, test_loader,
                 lr, num_epochs, num_launches), n_param, str(pruned_mlp)