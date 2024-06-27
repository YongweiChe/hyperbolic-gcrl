import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import hypll
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.optim import RiemannianAdam
import hypll.nn as hnn
from hypll.tensors import TangentTensor

import torch
import torch.nn as nn
import torch.nn.functional as F

def manifold_map(x, manifold):
    tangents = TangentTensor(x, man_dim=-1, manifold=manifold)
    return manifold.expmap(tangents)

class HyperbolicMLP(nn.Module):
    def __init__(self, in_features, out_features, manifold):
        super(HyperbolicMLP, self).__init__()
        self.manifold = manifold
        # Euclidean layers
        self.fc1_euc = nn.Linear(in_features, 64)
        self.fc2_euc = nn.Linear(64, 64)
        self.fc3_euc = nn.Linear(64, 32)

        # Hyperbolic layers
        self.fc1_hyp = hnn.HLinear(in_features=32, out_features=32, manifold=manifold)
        self.fc2_hyp = hnn.HLinear(in_features=32, out_features=16, manifold=manifold)
        self.fc3_hyp = hnn.HLinear(in_features=16, out_features=out_features, manifold=manifold)
        self.relu_hyp = hnn.HReLU(manifold=manifold)

    def forward(self, x):
        # Pass through Euclidean layers
        x = F.relu(self.fc1_euc(x))
        x = F.relu(self.fc2_euc(x))
        x = F.relu(self.fc3_euc(x))

        x = manifold_map(x, self.manifold)

        # Pass through Hyperbolic layers
        x = self.relu_hyp(self.fc1_hyp(x))
        x = self.relu_hyp(self.fc2_hyp(x))
        x = self.fc3_hyp(x)
        return x


def hyperbolic_infoNCE_loss(anchor, positive, negatives, temperature, manifold):
    positive_scores = -manifold.dist(x=anchor, y=positive).unsqueeze(1) / temperature
    negative_scores = -manifold.dist(x=anchor.unsqueeze(1), y=negatives) / temperature
    # print(f'positive scores shape: {positive_scores.shape}; negative_scores shape: {negative_scores.shape}')
 
    logits = torch.cat([positive_scores, negative_scores], dim=1)  # (batch_size, 1 + num_negatives)

    # Labels are all zeros because the positive examples are always in the first column
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    # Compute the cross-entropy loss between logits and labels
    return nn.CrossEntropyLoss()(logits, labels)


