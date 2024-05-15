import os
import torch
from torch import nn
import numpy as np
import random
from datasets import load_dataset
from torch.utils.data import DataLoader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_params(num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
               num_threads=NUM_THREADS, seed=SEED):

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    g = torch.Generator()
    g.manual_seed(seed)
    
    dataset = load_dataset("inria-soda/tabular-benchmark", data_files="reg_num/diamonds.csv")
    dataset = dataset['train'].with_format("torch").train_test_split(test_size=0.2)
    
    test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, num_workers=num_workers,
                                 worker_init_fn=seed_worker, generator=g)
    notest_dataset = dataset['train'].train_test_split(test_size=0.2)
    
    train_dataloader = DataLoader(notest_dataset['train'], batch_size=batch_size,
                                  num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    val_dataloader = DataLoader(notest_dataset['test'], batch_size=batch_size,
                                num_workers=num_workers, worker_init_fn=seed_worker, generator=g)

    return train_dataloader, val_dataloader, test_dataloader


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=GAIN)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)


def num_params(model):
    return sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())