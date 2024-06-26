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

from pyramid import create_pyramid
from continuous_maze import bfs, gen_traj, plot_traj, ContinuousGridEnvironment, TrajectoryDataset, LabelDataset
from hyperbolic_networks import HyperbolicMLP, hyperbolic_infoNCE_loss, manifold_map
from networks import StateActionEncoder, StateEncoder, infoNCE_loss

def evaluate(maze, num_trials, encoder1, encoder2, manifold, max_steps=100, hyperbolic=False, eps=10, device='cpu'):
    valid_indices = np.argwhere(maze == 0)
    np.random.shuffle(valid_indices)
    
    results = []
    for i in range(num_trials):
        with torch.no_grad():
            start, end = np.random.randint(0, len(valid_indices), size=2)
            start = tuple(valid_indices[start])
            end = tuple(valid_indices[end])
            
            goal = torch.tensor(end).to(torch.float32).to(device).unsqueeze(0)
            if hyperbolic:
                goal = manifold_map(goal, manifold=manifold)
            goal = encoder2(goal)
            
            env = ContinuousGridEnvironment(maze, start, {})
            
            cur_pos = env.agent_position
            # print(f'a: {cur_pos}')
            
            def reached(cur_pos, goal_pos):
                # print(f'cur pos: {cur_pos}')
                cur_pos = (int(cur_pos[0]), int(cur_pos[1]))
                goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
                return cur_pos == goal_pos
            
            def step():
                activations = []
                angles = torch.linspace(0., 2 * torch.pi, 16)
                for a in angles:
                    action = torch.tensor([torch.sin(a), torch.cos(a)])
                    cur = torch.tensor([cur_pos[0], cur_pos[1], torch.sin(a), torch.cos(a)]).to(torch.float32)
                    if hyperbolic:
                        cur = manifold_map(cur, manifold)
                    cur = encoder1(cur)

                    # MANIFOLD EVAL
                    if hyperbolic:
                        activations.append((action, -manifold.dist(x=cur, y=goal)))
                    else:
                        activations.append((action, -torch.norm(cur - goal)))

                best_action = activations[np.argmax([x[1].detach().numpy() for x in activations])][0]
                angle = np.arctan2(best_action[0], best_action[1]) + np.random.normal() * eps * (2 * np.pi / 360)
                best_action = torch.tensor(np.array([np.sin(angle), np.cos(angle)]))
                env.move_agent(best_action)
                # print(f'agent position: {env.agent_position}')
                
            steps = 0
            while not reached(env.agent_position, end):
                if steps > max_steps:
                    break
                step()
                steps += 1
            
            # print(reached(env.agent_position, end))
            # print(f'start: {start}, end: {end}, steps: {steps}')
            results.append((reached(env.agent_position, end), steps))
    return results
    
def train(num_epochs, maze, dataloader, encoder_mod1, encoder_mod2, optimizer, hyperbolic=False, manifold=None):
    print(f'is hyperbolic: {hyperbolic}')
    # Training loop
    accuracies = []
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for anchor, positive, negatives in dataloader:
            # (s,a) <-> (s)
            anchor = torch.tensor(anchor).to(torch.float32).to(device)
            positive = torch.tensor(positive).to(torch.float32).to(device)
            negatives = torch.tensor(negatives).to(torch.float32).to(device)
            
            if hyperbolic:
                m_anchor = manifold_map(anchor, manifold)
                m_positive = manifold_map(positive, manifold)
                m_negatives = manifold_map(negatives, manifold)
            else:
                m_anchor = anchor
                m_positive = positive
                m_negatives = negatives

            # print(f'negatives shape: {m_negatives.shape}')
            anchor_enc = encoder_mod1(m_anchor) # takes state, action tuple
            positive_enc = encoder_mod2(m_positive) # takes state
            negatives_enc = encoder_mod2(m_negatives)

            positive_action = anchor[:,[2,3]]
            cur_state = anchor[:,[0,1]]
            angle = torch.arctan2(anchor[:,2], anchor[:,3])

            negative_actions = (angle + torch.pi)[:,None] + (torch.rand(num_negatives)[None,:] - 0.5) * (3 * torch.pi / 2)
            negative_dirs = torch.stack([torch.sin(negative_actions), torch.cos(negative_actions)]).moveaxis(0, -1)
            # print(f'negative actions shape: {negative_actions.shape}')
            # print(negative_dirs.shape)
            negative_full = torch.cat((cur_state.unsqueeze(1).expand(-1, num_negatives, -1), negative_dirs), dim=-1).to(device)
            
            if hyperbolic:
                m_negative_full = manifold_map(negative_full, manifold)
            else:
                m_negative_full = negative_full

            # print(negative_full.shape)
            neg_action_enc = encoder_mod1(m_negative_full)
            # print(f'positive_enc: {positive_enc.shape}, anchor: {anchor_enc.shape}, neg_action_enc: {neg_action_enc.shape}')
            
            if hyperbolic:
                action_loss = hyperbolic_infoNCE_loss(positive_enc, anchor_enc, neg_action_enc, temperature, manifold=manifold)
                future_loss = hyperbolic_infoNCE_loss(anchor_enc, positive_enc, negatives_enc, temperature, manifold=manifold)
            else:
                action_loss = infoNCE_loss(positive_enc, anchor_enc, neg_action_enc, temperature, metric_type=1)
                future_loss = infoNCE_loss(anchor_enc, positive_enc, negatives_enc, temperature, metric_type=1)
            
            loss = action_loss + future_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss = total_loss / len(dataloader)
        evals = evaluate(maze, 25, encoder_mod1, encoder_mod2, manifold, max_steps=50, hyperbolic=hyperbolic)
        acc = np.mean([x[1] for x in evals])
        losses.append(loss)
        accuracies.append(acc)
        print(f'Epoch {epoch+1}, Loss: {loss}, Acc: {acc}')
    return losses, accuracies
        
