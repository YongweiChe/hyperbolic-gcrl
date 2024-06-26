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

class HyperbolicMLP(nn.Module):
    def __init__(self, in_features, out_features, manifold):
        super(HyperbolicMLP, self).__init__()
        self.fc1 = hnn.HLinear(in_features=in_features, out_features=64, manifold=manifold)
        self.fc2 = hnn.HLinear(in_features=64, out_features=64, manifold=manifold)
        self.fc3 = hnn.HLinear(in_features=64, out_features=out_features, manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def hyperbolic_infoNCE_loss(anchor, positive, negatives, temperature, manifold):
    positive_scores = -manifold.dist(x=anchor, y=positive).unsqueeze(1)
    negative_scores = -manifold.dist(x=anchor.unsqueeze(1), y=negatives)
    # print(f'positive scores shape: {positive_scores.shape}; negative_scores shape: {negative_scores.shape}')
 
    logits = torch.cat([positive_scores, negative_scores], dim=1)  # (batch_size, 1 + num_negatives)

    # Labels are all zeros because the positive examples are always in the first column
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    # Compute the cross-entropy loss between logits and labels
    return nn.CrossEntropyLoss()(logits, labels)

def manifold_map(x, manifold):
    tangents = TangentTensor(x, man_dim=-1, manifold=manifold)
    return manifold.expmap(tangents)
