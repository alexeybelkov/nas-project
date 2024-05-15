import torch
from torch import nn
import numpy as np
from copy import deepcopy
from scipy.stats import rankdata as rank


def calculate_k(loader, mlp, device, num_batch=1):

    assert num_batch > 0, "Number of used batches should be positive!"

    gradients = {}
    for edge in mlp.edges.keys():
        if len(mlp.edges[edge]) > 1:
            for ind in range(len(mlp.edges[edge])):
                gradients[(edge, ind)] = []

    mlp = mlp.to(device)

    for i, batch in enumerate(loader):

        if i >= num_batch:
            break

        batch = torch.stack(tuple(batch.values()),
                            dim=1)[:, :-1].cuda(device=device,
                                                non_blocking=True)
        for edge, ind in gradients:
            mlp.zero_grad()
            batch_ = batch.clone().cuda(device=device, non_blocking=True)

            mlp.delete(edge, ind)
            preds = mlp(batch_)
            mlp.restore()

            for j in range(len(preds)):
                preds[j:j + 1].backward(torch.ones_like(preds[j:j + 1]),
                                        retain_graph=True)
                grads = []
                for name, layer in mlp.named_parameters():
                    if 'weight' in name and layer.grad is not None:
                        grads.append(layer.grad.flatten().detach())
                gradients[(edge, ind)].append(torch.cat(grads, dim=-1))

                del grads
                mlp.zero_grad()
                torch.cuda.empty_cache()

            del batch_

    k = {}
    for edge, ind in gradients:
        grads = torch.stack(gradients[(edge, ind)], dim=0)
        ntk = torch.einsum('nc,mc->nm', [grads, grads])
        ev = torch.linalg.eigvalsh(ntk, UPLO='U')
        new_k = (ev[-1] / ev[0]).item()
        k[(edge, ind)] = - np.nan_to_num(new_k, copy=True, nan=1e7)

        del grads, ntk, ev

    return k


def calc_LR(activations):

    output = torch.matmul(activations.half(), (1 - activations).T.half())
    output = 1. / (torch.sum(1 - torch.sign(output + output.T), dim=1).float() + 1e-12)
    return round(output.sum().item())


def calculate_lr(loader, mlp, device, num_batch=1):

    assert num_batch > 0, "Number of used batches should be positive!"

    lrs = {}
    for edge in mlp.edges.keys():
        if len(mlp.edges[edge]) > 1:
            for ind in range(len(mlp.edges[edge])):
                lrs[(edge, ind)] = []

    for edge, ind in lrs:

        ptr = 0
        model = deepcopy(mlp).to(device)
        model.delete(edge, ind)
        LR = 0

        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(
                    hook=lambda mod, inp, out: iFeats.append(out.detach())
                )

        for i, batch in enumerate(loader):

            if i >= num_batch:
                break

            iFeats = []

            batch = torch.stack(tuple(batch.values()),
                                dim=1)[:, :-1].cuda(device=device,
                                                    non_blocking=True)
            batch_size_ = len(batch)
            model.zero_grad()
            interFeatures = []

            with torch.no_grad():
                model.forward(batch)

            if len(iFeats) == 0:
                continue

            activations = torch.cat([feat.view(batch_size_, -1)
                                     for feat in iFeats], dim=1)

            LR += calc_LR(torch.sign(activations))

            ptr += batch_size_

            del iFeats
            del activations

        del model

        torch.cuda.empty_cache()
        lrs[(edge, ind)] = -LR

    return lrs


def TENAS(loader, mlp, device, num_batch=1):

  for cur_size in range(SPACE_SIZE, 1, -1):
    k = calculate_k(loader, mlp, device, num_batch)
    lr = calculate_lr(loader, mlp, device, num_batch)

    k_list, lr_list = [], []

    for edge in mlp.edges:
      for ind in range(len(mlp.edges[edge])):
          k_list.append(-k[(edge, ind)])
          lr_list.append(lr[(edge, ind)])

    k_ranks = rank(k_list)
    lr_ranks = rank(lr_list)
    total_ranks = k_ranks + lr_ranks

    ptr = 0
    for edge in mlp.edges:
      ind = np.argmin(total_ranks[ptr:ptr + cur_size])
      mlp.perm_delete(edge, ind)
      ptr += cur_size

  return mlp
