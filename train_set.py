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
from matplotlib.animation import FuncAnimation
import argparse
import yaml
import math
from pyramid import create_pyramid
from continuous_maze import bfs, get_trajectories, gen_traj, plot_traj, ContinuousGridEnvironment, TrajectoryDataset, LabelDataset
from hyperbolic_networks import HyperbolicMLP, hyperbolic_infoNCE_loss, manifold_map
from networks import StateActionEncoder, StateEncoder, infoNCE_loss
import os
import time
from train_streets import get_maze
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"


import torch
import torch.nn.functional as F

def contrastive_loss(feature1_embed, feature2_embed, temperature):
    # Compute pairwise L2 distances
    dist_matrix = torch.cdist(feature1_embed, feature2_embed, p=2.0)
    
    # Convert distances to similarities (negative distances)
    logits = -dist_matrix / temperature
    
    labels = torch.arange(feature1_embed.shape[0], device=feature1_embed.device)
    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.t(), labels)
    return (loss_1 + loss_2) / 2

class PointCloudEncoder(nn.Module):
    def __init__(self, input_dim=8, output_dim=256):
        super(PointCloudEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.linear1 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # x shape: (batch_size, n, 8)
        x = x.transpose(2, 1)  # (batch_size, 8, n)
        
        # Point-wise MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.linear1(x)))
        x = self.linear2(x)
        
        return x
    


def pad_array(arr, target_length=100):
    """
    Pad an nx2 numpy array to target_length x 2 with zeros.
    
    Args:
    - arr: Input numpy array of shape (n, 2)
    - target_length: Desired length of the first dimension (default: 100)
    
    Returns:
    - Padded numpy array of shape (target_length, 2)
    """
    current_length = arr.shape[0]
    
    if current_length >= target_length:
        # If the array is already long enough, return the first target_length rows
        return arr[:target_length]
    
    # Calculate the amount of padding needed
    pad_length = target_length - current_length
    
    # Create the padding
    padding = ((0, pad_length), (0, 0))
    
    # Pad the array
    padded_arr = np.pad(arr, padding, mode='constant', constant_values=0)
    
    return padded_arr


class SetDataset(Dataset):
    def __init__(self, maze, num_trajectories, embedding_dim=2, num_negatives=10, gamma=0.1, order_fn=None, num_splits=4, padding_len=100):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.num_negatives = num_negatives
        self.num_splits = num_splits
        self.padding_len = padding_len
        self.maze = maze
        self.gamma = gamma
        print(f'gamma: {self.gamma}')

        self.trajectories = get_trajectories(maze, num_trajectories, order_fn=order_fn)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # Anchor: the current data point
        
        traj = self.trajectories[idx]
        split_traj = np.split(traj, np.random.randint(0, len(traj), size=self.num_splits))
        split_traj = list(filter(lambda x: x.shape[0] != 0,split_traj))
        # print(split_traj)
        
        i, j = np.random.randint(0, len(split_traj), 2)
        # print(i, j)
        set1 = pad_array(np.stack([x[0] for x in split_traj[i]]), target_length=self.padding_len)
        set2 = pad_array(np.stack([x[0] for x in split_traj[j]]), target_length=self.padding_len)
        
        return set1, set2


def save_models(encoder1, encoder2,epoch, name=''):
    os.makedirs('models', exist_ok=True)
    torch.save(encoder1.state_dict(), f'models/{name}_encoder1_epoch_{epoch}.pth')
    torch.save(encoder2.state_dict(), f'models/{name}_encoder2_epoch_{epoch}.pth')


# @profile
def main():
    parser = argparse.ArgumentParser(description='Run experiment with config file.')
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to the YAML config file')
    parser.add_argument('--temperature', type=float, help='Temperature value to override the config file')
    args = parser.parse_args()

    # Load the config file
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Override temperature if provided as an argument
    if args.temperature is not None:
        config['temperature'] = args.temperature

    maze = get_maze(config['maze_type'])
    experiment_name = f"set_experiment{config['custom']}_hyperbolic_{config['hyperbolic']}_epochs_{config['num_epochs']}_temp_{config['temperature']}_trajectories_{config['num_trajectories']}_order_{config['order_name']}_maze_{config['maze_type']}_embeddingdim_{config['embedding_dim']}_gamma_{config['gamma']}_batch_{config['batch_size']}_hyp_layers_{config['hyp_layers']}"

    # Initialize wandb
    wandb.init(
        project=config['project'],
        name=experiment_name,
        config={
            **config
        }
    )

    # configs
    config = wandb.config

    manifold = PoincareBall(c=Curvature(value=0.1, requires_grad=True))
    dataset = SetDataset(maze, 
                        config.num_trajectories, 
                        embedding_dim=config.embedding_dim, 
                        num_negatives=config.num_negatives, 
                        gamma=config.gamma,
                        order_fn=None,
                        num_splits=4
                        )
    
    dataloader = DataLoader(dataset, 
                            batch_size=config.batch_size, 
                            shuffle=True, 
                            num_workers=config.num_workers, 
                            pin_memory=True)
    
    encoder1 = PointCloudEncoder(input_dim=2, output_dim=config.embedding_dim).to(device)
    encoder2 = PointCloudEncoder(input_dim=2, output_dim=config.embedding_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder1.parameters()) + list(encoder2.parameters()), lr=config.learning_rate)

    # Training loop
    total_batches = 0
    start_time = time.time()

    for epoch in range(config.num_epochs):
        total_loss = 0

        for s1, s2 in dataloader:
            s1 = torch.as_tensor(s1, dtype=torch.float32, device=device)
            s2 = torch.as_tensor(s2, dtype=torch.float32, device=device)
            # print(f's1 shape: {s1.shape}, s2 shape: {s2.shape}')

            s1_enc = encoder1(s1)
            s2_enc = encoder2(s2)

            loss = contrastive_loss(s1_enc, s2_enc, config.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            total_batches += 1
            
            if total_batches % 100 == 0:
                elapsed_time = time.time() - start_time
                batches_per_second = total_batches / elapsed_time
                print(f"Epoch {epoch}, Batch {total_batches}: {batches_per_second:.2f} batches/second")

        epoch_time = time.time() - start_time
        epoch_batches_per_second = total_batches / epoch_time
        print(f"Epoch {epoch} complete. Average: {epoch_batches_per_second:.2f} batches/second")

        loss = total_loss / len(dataloader)

        metrics = {
                    "epoch": epoch + 1,
                    "loss": loss
                }
        wandb.log(metrics)  
        print(f'Epoch {epoch+1}, Loss: {loss}')

        if epoch % 32 == 0 and epoch != 0:
            save_models(encoder1, encoder2, epoch + 1, experiment_name)
    save_models(encoder1, encoder2, epoch + 1, experiment_name)

if __name__ == '__main__':
    main()




