import torch
from torch import nn


class Zero(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, batch):
        return torch.zeros_like(batch)
    

class ClassicMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), # 0 -> 1
            nn.Linear(hidden_dim, hidden_dim), # 1 -> 2
            nn.ReLU(), # 2 -> 3
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):

        return self.nn(batch)
    

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.in_ = nn.Linear(input_dim, hidden_dim)
        self.out_ = nn.Linear(hidden_dim, 1)

        self.edges = nn.ModuleDict({
            '1': get_modules(hidden_dim, hidden_dim),
            '2': get_modules(hidden_dim, hidden_dim),
            '3': get_modules(hidden_dim, hidden_dim),
            '4': get_modules(hidden_dim, hidden_dim),
            '5': get_modules(hidden_dim, hidden_dim),
            '6': get_modules(hidden_dim, hidden_dim),
        })

        self.saved = None

    def perm_delete(self, edge, ind):

        self.edges[edge].pop(ind)


    def delete(self, edge, ind):

        if edge is None or ind is None:
            return

        self.saved = (edge, deepcopy(self.edges[edge]))
        if len(self.edges[edge]) > 1:
            self.edges[edge].pop(ind)

    def restore(self):
        if self.saved is not None:
            self.edges[self.saved[0]] = self.saved[1]
            self.saved = None

    def forward(self, batch):

        node0 = self.in_(batch)

        edge1 = torch.stack([op(node0) for op in self.edges['1']], dim=1).sum(1)

        node1 = edge1

        edge2 = torch.stack([op(node0) for op in self.edges['2']], dim=1).sum(1)
        edge3 = torch.stack([op(node0) for op in self.edges['3']], dim=1).sum(1)

        edge4 = torch.stack([op(node1) for op in self.edges['4']], dim=1).sum(1)
        edge5 = torch.stack([op(node1) for op in self.edges['5']], dim=1).sum(1)

        node2 = edge2 + edge4

        edge6 = torch.stack([op(node2) for op in self.edges['6']], dim=1).sum(1)

        node3 = edge3 + edge5 + edge6

        return self.out_(node3)


def get_modules(input_dim, output_dim):

    return nn.ModuleList([
        Zero(), nn.Identity(), nn.ReLU(),
        nn.Sigmoid(), nn.Linear(input_dim, output_dim)
    ])