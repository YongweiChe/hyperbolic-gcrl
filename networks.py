import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from continuous_maze import bfs, gen_traj, plot_traj, ContinuousGridEnvironment, TrajectoryDataset, LabelDataset

class FlexibleCNN(nn.Module):
    def __init__(self, input_channels, out_features, sample_input):
        super(FlexibleCNN, self).__init__()
        self.output_dim = out_features

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the output from the conv layers
        self._conv_output_size = self._get_conv_output_size(sample_input.shape[1:])
        self.fc1 = nn.Linear(self._conv_output_size, self.output_dim)

        

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
            dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
            return int(torch.flatten(dummy_output, 1).size(1))

    def forward(self, x):
        # Convolutional layer 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Convolutional layer 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc1(x)
        
        return x


class ActionNetwork(nn.Module):
    def __init__(self, image_encoder, embedding_dim):
        super(ActionNetwork, self).__init__()
        self.image_encoder = image_encoder
        self.intermed_dim = self.image_encoder.output_dim
        self.fc1 = nn.Linear(2, self.intermed_dim)
        self.fc2 = nn.Linear(2 * self.intermed_dim, 2 * self.intermed_dim)
        self.fc3 = nn.Linear(2 * self.intermed_dim, self.intermed_dim)
        self.fc4 = nn.Linear(self.intermed_dim, embedding_dim)

    def forward(self, image, action): # action should be 2-dimensional
        x1 = self.image_encoder(image)
        x2 = self.fc1(action)
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the StateActionEncoder
class StateActionEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(StateActionEncoder, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)  # New layer
        self.fc4 = nn.Linear(64, embedding_dim)

    def forward(self, state_action):
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # New layer
        x = self.fc4(x)
        return x

# Define the StateEncoder
class StateEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(StateEncoder, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)  # New layer
        self.fc4 = nn.Linear(64, embedding_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # New layer
        x = self.fc4(x)
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
