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


"""
Hyperbolic Networks!
"""

def manifold_map(x, manifold):
    """
    Maps a tensor in Euclidean space onto a Riemannian Manifold
    """
    tangents = TangentTensor(x, man_dim=-1, manifold=manifold)
    return manifold.expmap(tangents)

class HyperbolicMLP(nn.Module):
    """
    MLP but outputs a vector on a Riemannian Manifold
    in_features: input dimension
    manifold: Desired manifold (e.g. Poincare Disc)
    euc_width: How long the euclidean layers are
    hyp_widths: how long the hyperbolic layers are
    """
    def __init__(self, in_features, manifold, euc_widths, hyp_widths):
        super(HyperbolicMLP, self).__init__()
        self.manifold = manifold

        # Euclidean layers
        euc_layers = []
        prev_width = in_features
        for width in euc_widths:
            euc_layers.append(nn.Linear(prev_width, width))
            euc_layers.append(nn.ReLU())
            prev_width = width
        self.euc_layers = nn.Sequential(*euc_layers)

        # Hyperbolic layers
        hyp_layers = []
        prev_width = euc_widths[-1]
        for i, width in enumerate(hyp_widths):
            hyp_layers.append(hnn.HLinear(in_features=prev_width, out_features=width, manifold=manifold))
            if i < len(hyp_widths) - 1:  # Don't add ReLU after the last layer
                hyp_layers.append(hnn.HReLU(manifold=manifold))
            prev_width = width
        self.hyp_layers = nn.ModuleList(hyp_layers)

    def forward(self, x):
        # Pass through Euclidean layers
        x = self.euc_layers(x)

        x = manifold_map(x, self.manifold)

        # Pass through Hyperbolic layers
        for layer in self.hyp_layers:
            x = layer(x)
        return x

def hyperbolic_infoNCE_loss(anchor, positive, negatives, temperature, manifold):
    """
    InfoNCE but with manifold dist.
    TODO: Can probably coalesce with the euclidean infoNCE with a conditional branch
    """
    positive_scores = -manifold.dist(x=anchor, y=positive).unsqueeze(1) / temperature
    negative_scores = -manifold.dist(x=anchor.unsqueeze(1), y=negatives) / temperature
    # print(f'positive scores shape: {positive_scores.shape}; negative_scores shape: {negative_scores.shape}')
 
    logits = torch.cat([positive_scores, negative_scores], dim=1)  # (batch_size, 1 + num_negatives)

    # Labels are all zeros because the positive examples are always in the first column
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    # Compute the cross-entropy loss between logits and labels
    return nn.CrossEntropyLoss()(logits, labels)


