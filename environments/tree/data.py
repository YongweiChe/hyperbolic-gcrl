import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from environments.tree.tree import NaryTreeEnvironment
import math
import random

def get_tree_trajectories(tree, num_trajectories):
    l = []

    for _ in range(num_trajectories):
        start_node = np.random.randint(0, tree.num_nodes)
        end_node = np.random.randint(0, tree.num_nodes)
        l.append(tree.get_action_path(start_node, end_node))

    return l


def get_maze_trajectories(maze_env, num_trajectories):
    trajectories = []

    for _ in range(num_trajectories):
        # Get all valid positions (non-wall cells)
        valid_positions = [maze_env.flatten_state(pos) for pos in zip(*np.where(maze_env.maze != 1))]
        
        # Randomly choose start and end positions
        start_pos = random.choice(valid_positions)
        end_pos = random.choice(valid_positions)
        
        # Get the action path between these positions
        action_path = maze_env.get_action_path(start_pos, end_pos)
        
        # If a valid path exists, add it to the trajectories
        if action_path:
            trajectories.append(action_path)
        
    return trajectories



import numpy as np
from torch.utils.data import Dataset

class GeneralTrajectoryDataset(Dataset):
    def __init__(self, trajectories, valid_indices, num_negatives=10, gamma=0.1):
        super().__init__()
        self.trajectories = trajectories
        self.num_trajectories = len(trajectories)
        self.num_negatives = num_negatives
        self.gamma = gamma
        self.valid_indices = set(valid_indices)  # Convert to set for faster lookup
        print(f'gamma: {self.gamma}')

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        start = np.random.randint(0, len(traj))
        end = min(start + np.random.geometric(p=self.gamma), len(traj) - 1)

        anchor = np.array(traj[start])  # Keep as state-action pair
        positive_example = traj[end][0]  # Just the state for positive example

        # Use valid_indices to determine valid negative examples
        traj_states = set(state for state, _ in traj)
        valid_negatives = list(self.valid_indices - traj_states)
        
        negative_examples = np.random.choice(valid_negatives, size=self.num_negatives, replace=True)

        return anchor, np.array([positive_example]), negative_examples[:, None]
    

class TrajectoryDataset(Dataset):
    def __init__(self, depth, branching_factor, num_trajectories, num_negatives=10, gamma=0.1, order_fn=None):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.num_negatives = num_negatives
        self.tree = NaryTreeEnvironment(depth=depth, branching_factor=branching_factor)
        self.gamma = gamma
        print(f'gamma: {self.gamma}')

        self.trajectories = get_tree_trajectories(self.tree, num_trajectories)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        # Anchor: the current data point
        
        traj = self.trajectories[idx]
        # print(traj)
        # start, end = np.sort(np.random.randint(0, len(traj), size=2))
        start = np.random.randint(0, len(traj))
        end = min(start + np.random.geometric(p=self.gamma), len(traj) - 1)

        anchor = np.array(self.trajectories[idx][start])
        positive_example = self.trajectories[idx][end][0]

        negative_examples = []

        valid_negatives = list(set(np.arange(self.tree.num_nodes)) - set([x[0] for x in traj]))

        for i in range(self.num_negatives):
          idy = np.random.randint(0, len(valid_negatives))
          neg_state = valid_negatives[idy]
          negative_examples.append(neg_state)

        return anchor, np.array([positive_example]), np.array(negative_examples)[:,None]

class SetDataset(Dataset):
    """
    Returns positive pairs of set-valued outputs from the same trajectory
    """
    def __init__(self, depth, branching_factor, num_trajectories, num_negatives=10, gamma=0.1, order_fn=None, num_splits=4):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.num_negatives = num_negatives
        self.tree = NaryTreeEnvironment(depth=depth, branching_factor=branching_factor)
        self.gamma = gamma
        self.num_splits=num_splits
        print(f'set dataset gamma: {self.gamma}')

        self.trajectories = get_tree_trajectories(self.tree, num_trajectories)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = np.array(self.trajectories[idx])
        traj_len = len(traj)

        if traj_len == 1:
            log_size = 0
        else:
            log_size = min(np.random.geometric(p=self.gamma) - 1, math.floor(math.log2(traj_len)))

        size1 = int(math.pow(2, log_size))
        size2 = int(math.pow(2, max(log_size - 1, 0)))  # Keep the original size2 calculation

        # Decide whether set2 comes before or after set1
        before = np.random.choice([True, False])

        if before:
            # set2 comes before set1
            i2 = max(0, traj_len - size1 - size2)  # Ensure we have enough space for both sets
            i1 = i2 + size2

            if i1 + size1 >= traj_len:
                i1 = i2
        else:
            # set2 comes after set1
            i1 = np.random.randint(0, max(traj_len - size1 - size2 + 1, 1))
            i2 = i1 + size1

            if i2 + size2 >= traj_len:
                i2 = i1
        
        set1 = traj[i1:(i1 + size1), 0]
        set2 = traj[i2:(i2 + size2), 0]

        return set1[:, None], set2[:, None]