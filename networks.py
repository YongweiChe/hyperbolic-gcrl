import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from continuous_maze import bfs, gen_traj, plot_traj, ContinuousGridEnvironment, TrajectoryDataset, LabelDataset


# Define the StateActionEncoder
class StateActionEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(StateActionEncoder, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # Increased from 128 to 256
        self.fc2 = nn.Linear(64, 64)  # Added an extra layer
        self.fc3 = nn.Linear(64, embedding_dim)  # Increased from 128 to 512, then to embedding_dim

    def forward(self, state_action):
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the StateEncoder
class StateEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(StateEncoder, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Increased from 128 to 256
        self.fc2 = nn.Linear(64, 64)  # Added an extra layer
        self.fc3 = nn.Linear(64, embedding_dim)  # Increased from 128 to 512, then to embedding_dim

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CategoricalEncoder(nn.Module):
    def __init__(self, num_categories, embedding_dim):
        super(CategoricalEncoder, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, embedding_dim)

    def forward(self, categorical_data):
        embedded = self.embedding(categorical_data)
        # print(embedded)
        x = F.relu(self.fc1(embedded))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def infoNCE_loss(anchor, positive, negatives, temperature, metric_type=1):
    if metric_type == 1:
        # Compute negative L2 distance for the positive pairs
        positive_distances = torch.sum((anchor - positive) ** 2, dim=1, keepdim=True)
        positive_scores = -positive_distances / temperature

        # Compute negative L2 distances for the negative pairs
        negative_distances = torch.sum((anchor.unsqueeze(1) - negatives) ** 2, dim=2)
        negative_scores = -negative_distances / temperature
    else:
        # Use dot product for the positive and negative pairs (default)
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        positive_scores = torch.sum(anchor * positive, dim=1, keepdim=True) / temperature
        negative_scores = torch.matmul(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / temperature

    # print(positive_scores.shape)
    # print(negative_scores.shape)
    # Combine positive and negative scores into logits
    logits = torch.cat([positive_scores, negative_scores], dim=1)  # (batch_size, 1 + num_negatives)

    # Labels are all zeros because the positive examples are always in the first column
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    # Compute the cross-entropy loss between logits and labels
    return nn.CrossEntropyLoss()(logits, labels)
