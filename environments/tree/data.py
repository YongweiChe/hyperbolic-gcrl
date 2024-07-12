import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from environments.tree.tree import NaryTreeEnvironment


def get_trajectories(tree, num_trajectories):
    l = []

    for _ in range(num_trajectories):
        start_node = np.random.randint(0, tree.num_nodes)
        end_node = np.random.randint(0, tree.num_nodes)
        l.append(tree.get_action_path(start_node, end_node))

    return l

class TrajectoryDataset(Dataset):
    def __init__(self, depth, branching_factor, num_trajectories, num_negatives=10, gamma=0.1, order_fn=None):
        super().__init__()
        self.num_trajectories = num_trajectories
        self.num_negatives = num_negatives
        self.tree = NaryTreeEnvironment(depth=depth, branching_factor=branching_factor)
        self.gamma = gamma
        print(f'gamma: {self.gamma}')

        self.trajectories = get_trajectories(self.tree, num_trajectories)

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
        print(f'gamma: {self.gamma}')

        self.trajectories = get_trajectories(self.tree, num_trajectories)

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

        return set1[:,None], set2[:,None]