import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
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
from continuous_maze import (
    bfs,
    get_trajectories,
    gen_traj,
    plot_traj,
    ContinuousGridEnvironment,
    TrajectoryDataset,
    LabelDataset,
)
from hyperbolic_networks import HyperbolicMLP, hyperbolic_infoNCE_loss, manifold_map
from networks import StateActionEncoder, StateEncoder, infoNCE_loss
import os
import time
from train_streets import get_maze
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"


def contrastive_loss(
    feature1_embed, feature2_embed, temperature, hyperbolic=False, manifold=None
):
    # Compute pairwise L2 distances
    if hyperbolic:
        dist_matrix = manifold.dist(
            x=feature1_embed.unsqueeze(1), y=feature2_embed.unsqueeze(0)
        )
    else:
        dist_matrix = torch.cdist(feature1_embed, feature2_embed, p=2.0)

    # Convert distances to similarities (negative distances)
    logits = -dist_matrix / temperature

    labels = torch.arange(feature1_embed.shape[0], device=device)
    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.t(), labels)
    return (loss_1 + loss_2) / 2


class DeepSet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepSet, self).__init__()

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Correct calculation for upper triangle size
        cov_dim = (hidden_dim * (hidden_dim + 1)) // 2
        
        # Separate network for processing covariance
        self.cov_network = nn.Sequential(
            nn.Linear(cov_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Now 2 * hidden_dim due to mean and processed covariance
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, mask):
        # x shape: (batch_size, max_set_size, input_dim)
        # mask shape: (batch_size, max_set_size)
        # print(f'x shape: {x.shape}, mask shape: {mask.shape}')
        # Apply phi to each element
        x = self.phi(x)  # Shape: (batch_size, max_set_size, hidden_dim)

        # Apply mask to zero out padding
        x = x * mask.unsqueeze(-1)

        # Compute mean
        x_sum = torch.sum(x, dim=1)
        set_sizes = torch.sum(mask, dim=1, keepdim=True)
        x_mean = x_sum / set_sizes  # Shape: (batch_size, hidden_dim)

        # Compute covariance
        x_centered = x - x_mean.unsqueeze(1)
        x_centered = x_centered * mask.unsqueeze(-1)  # Apply mask again
        cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / (
            set_sizes - 1
        ).unsqueeze(-1).clamp(min=1)

        # Extract upper triangular part of covariance (including diagonal)
        batch_size, hidden_dim, _ = cov.shape
        triu_indices = torch.triu_indices(hidden_dim, hidden_dim)
        cov_vector = cov[:, triu_indices[0], triu_indices[1]]

        # Process covariance vector through its own network
        processed_cov = self.cov_network(cov_vector)  # Shape: (batch_size, hidden_dim)

        # Concatenate mean and processed covariance information
        x_combined = torch.cat([x_mean, processed_cov], dim=1)

        # Apply rho to the combined representation
        output = self.rho(x_combined)

        return output



class HyperbolicDeepSet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, manifold):
        super(HyperbolicDeepSet, self).__init__()

        self.deepset = DeepSet(input_dim, hidden_dim, hidden_dim)
        self.manifold = manifold

        # Hyperbolic layers
        self.hyp_layer1 = hnn.HLinear(
            in_features=hidden_dim, out_features=hidden_dim, manifold=manifold
        )
        self.hyp_relu = hnn.HReLU(manifold=manifold)
        self.hyp_layer2 = hnn.HLinear(
            in_features=hidden_dim, out_features=output_dim, manifold=manifold
        )

    def forward(self, x, mask):
        # Process input through DeepSet
        x = self.deepset(x, mask)

        # Map directly to hyperbolic space
        x = manifold_map(x, self.manifold)

        # Pass through Hyperbolic layers
        x = self.hyp_layer1(x)
        x = self.hyp_relu(x)
        x = self.hyp_layer2(x)

        return x


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

    def forward(self, x, mask):
        # x shape: (batch_size, n, 8)
        x = x * mask.unsqueeze(-1)
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


def collate_fn(batch):
    # Separate set1 and set2
    set1_list, set2_list = zip(*batch)

    # Convert to tensors
    set1_tensors = [torch.tensor(s, dtype=torch.float) for s in set1_list]
    set2_tensors = [torch.tensor(s, dtype=torch.float) for s in set2_list]

    # Pad sequences
    set1_padded = pad_sequence(set1_tensors, batch_first=True)
    set2_padded = pad_sequence(set2_tensors, batch_first=True)

    # Get lengths
    set1_lengths = torch.tensor([len(s) for s in set1_list])
    set2_lengths = torch.tensor([len(s) for s in set2_list])

    # Create masks
    batch_size, max_set_size = set1_padded.shape[:2]
    set1_mask = torch.arange(max_set_size).expand(batch_size, max_set_size) < set1_lengths.unsqueeze(1)
    
    batch_size, max_set_size = set2_padded.shape[:2]
    set2_mask = torch.arange(max_set_size).expand(batch_size, max_set_size) < set2_lengths.unsqueeze(1)

    return {
        'set1': set1_padded,
        'set2': set2_padded,
        'set1_mask': set1_mask.float(),
        'set2_mask': set2_mask.float(),
        'set1_lengths': set1_lengths,
        'set2_lengths': set2_lengths
    }

class SetDataset(Dataset):
    def __init__(
        self,
        maze,
        num_trajectories,
        embedding_dim=2,
        num_negatives=10,
        gamma=0.1,
        order_fn=None,
        num_splits=4,
        padding_len=100,
    ):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.num_negatives = num_negatives
        self.num_splits = num_splits
        self.padding_len = padding_len
        self.maze = maze
        self.gamma = gamma
        print(f"gamma: {self.gamma}")

        self.trajectories = get_trajectories(maze, num_trajectories, order_fn=order_fn)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # Anchor: the current data point

        traj = self.trajectories[idx]
        split_traj = np.split(
            traj, np.random.randint(0, len(traj), size=self.num_splits)
        )
        split_traj = list(filter(lambda x: x.shape[0] != 0, split_traj))
        # print(split_traj)

        i, j = np.random.randint(0, len(split_traj), 2)
        # print(i, j)

        set1 = np.stack([x[0] for x in split_traj[i]])
        set2 = np.stack([x[0] for x in split_traj[j]])

        return set1, set2


def save_models(encoder1, encoder2, epoch, name=""):
    os.makedirs("models", exist_ok=True)
    torch.save(encoder1.state_dict(), f"models/{name}_encoder1_epoch_{epoch}.pth")
    torch.save(encoder2.state_dict(), f"models/{name}_encoder2_epoch_{epoch}.pth")


# @profile
def main():
    parser = argparse.ArgumentParser(description="Run experiment with config file.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature value to override the config file",
    )
    args = parser.parse_args()

    # Load the config file
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    # Override temperature if provided as an argument
    if args.temperature is not None:
        config["temperature"] = args.temperature

    maze = get_maze(config["maze_type"])
    experiment_name = f"set_experiment{config['custom']}_hyperbolic_{config['hyperbolic']}_epochs_{config['num_epochs']}_temp_{config['temperature']}_trajectories_{config['num_trajectories']}_order_{config['order_name']}_maze_{config['maze_type']}_embeddingdim_{config['embedding_dim']}_gamma_{config['gamma']}_batch_{config['batch_size']}_hyp_layers_{config['hyp_layers']}"

    # Initialize wandb
    wandb.init(project=config["project"], name=experiment_name, config={**config})

    # configs
    config = wandb.config

    manifold = PoincareBall(c=Curvature(value=1.0, requires_grad=False)) # requires_grad = True sets learnable curvature parameter
    dataset = SetDataset(
        maze,
        config.num_trajectories,
        embedding_dim=config.embedding_dim,
        num_negatives=config.num_negatives,
        gamma=config.gamma,
        order_fn=None,
        num_splits=4,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    if config.hyperbolic:
        encoder1 = HyperbolicDeepSet(input_dim=2, hidden_dim=64, output_dim=config.embedding_dim, manifold=manifold).to(device)
        # encoder2 = HyperbolicDeepSet(input_dim=2, hidden_dim=64, output_dim=config.embedding_dim, manifold=manifold).to(device)
        optimizer = RiemannianAdam(
            list(encoder1.parameters()), #  + list(encoder2.parameters()
            lr=config.learning_rate,
        )
    else:
        PC = True
        encoder1 = DeepSet(input_dim=2, hidden_dim=64, output_dim=config.embedding_dim).to(device)
        if PC:
            encoder1 = PointCloudEncoder(input_dim=2, output_dim=config.embedding_dim).to(device)
        # encoder2 = DeepSet(input_dim=2, hidden_dim=64, output_dim=config.embedding_dim).to(device)
        optimizer = torch.optim.Adam(
            list(encoder1.parameters()), #  + list(encoder2.parameters()
            lr=config.learning_rate,
        )
    
    print(encoder1)

    # Training loop
    total_batches = 0
    start_time = time.time()

    for epoch in range(config.num_epochs):
        total_loss = 0

        for batch in dataloader:
            s1 = torch.as_tensor(batch['set1'], dtype=torch.float32, device=device)
            s2 = torch.as_tensor(batch['set2'], dtype=torch.float32, device=device)

            mask1 = torch.as_tensor(batch['set1_mask'], dtype=torch.float32, device=device)
            mask2 = torch.as_tensor(batch['set2_mask'], dtype=torch.float32, device=device)
            # print(f's1 shape: {s1.shape}, s2 shape: {s2.shape}')

            s1_enc = encoder1(s1, mask1)
            s2_enc = encoder1(s2, mask2)

            loss = contrastive_loss(s1_enc, s2_enc, config.temperature, hyperbolic=config.hyperbolic, manifold=manifold)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            total_batches += 1

            if total_batches % 100 == 0:
                elapsed_time = time.time() - start_time
                batches_per_second = total_batches / elapsed_time
                print(
                    f"Epoch {epoch}, Batch {total_batches}: {batches_per_second:.2f} batches/second"
                )

        epoch_time = time.time() - start_time
        epoch_batches_per_second = total_batches / epoch_time
        print(
            f"Epoch {epoch} complete. Average: {epoch_batches_per_second:.2f} batches/second"
        )

        loss = total_loss / len(dataloader)

        metrics = {"epoch": epoch + 1, "loss": loss}
        wandb.log(metrics)
        print(f"Epoch {epoch+1}, Loss: {loss}")

        if epoch % 32 == 0 and epoch != 0:
            save_models(encoder1, encoder1, epoch + 1, experiment_name) # CHANGE THIS ONCE TWO ENCODERS ARE
    save_models(encoder1, encoder1, epoch + 1, experiment_name)


if __name__ == "__main__":
    main()
