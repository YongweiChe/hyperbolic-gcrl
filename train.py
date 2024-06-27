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
from pyramid import create_pyramid
from continuous_maze import bfs, gen_traj, plot_traj, ContinuousGridEnvironment, TrajectoryDataset, LabelDataset
from hyperbolic_networks import HyperbolicMLP, hyperbolic_infoNCE_loss, manifold_map
from networks import StateActionEncoder, StateEncoder, infoNCE_loss


import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(maze, num_trials, encoder1, encoder2, manifold, max_steps=100, hyperbolic=False, eps=10., step_size=0.5, verbose=False):
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
            
            # print(start)
            env = ContinuousGridEnvironment(maze, start, {})
            
            def reached(cur_pos, goal_pos):
                # print(f'cur pos: {cur_pos}')
                cur_pos = (int(cur_pos[0]), int(cur_pos[1]))
                goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
                return cur_pos == goal_pos
            
            def step():
                cur_pos = env.agent_position
                if verbose:
                    print(f'cur_pos: {cur_pos}, goal: {goal}')
                activations = []
                angles = torch.linspace(0., 2 * torch.pi, 16)
                for a in angles:
                    action = torch.tensor([torch.sin(a), torch.cos(a)])
                    cur = torch.tensor([cur_pos[0], cur_pos[1], torch.sin(a), torch.cos(a)]).to(device, torch.float32)
                    if hyperbolic:
                        cur = manifold_map(cur, manifold)
                    cur = encoder1(cur)

                    # MANIFOLD EVAL
                    if hyperbolic:
                        activations.append((action, -manifold.dist(x=cur, y=goal)))
                    else:
                        activations.append((action, -torch.norm(cur - goal)))
                        
            

                best_action = activations[np.argmax([x[1].cpu() for x in activations])][0]
                angle = np.arctan2(best_action[0], best_action[1]) + np.random.normal() * eps * (2 * np.pi / 360)
                best_action = torch.tensor(np.array([np.sin(angle), np.cos(angle)]))
                env.move_agent(best_action)
                # print(f'agent position: {env.agent_position}')
                
                
            def SPL(maze, start, end, num_steps, success): # Success weighted by (normalized inverse) Path Length
                if not success:
                    return 0
                else:
                    p = num_steps * step_size
                    l = len(bfs(maze, start, end))
                    return (l / max(p, l))
            
            steps = 0
            while not reached(env.agent_position, end):
                if steps > max_steps:
                    break
                step()
                steps += 1
                
            result = (not reached(env.agent_position, end), steps, SPL(maze, start, end, steps, reached(env.agent_position, end)))
            if verbose:
                print(reached(env.agent_position, end))
                print(f'start: {start}, goal: {end}, end_pos: {env.agent_position}, steps: {steps}')
                print(results)
                
            results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Maze experiment parameters.')
    parser.add_argument('--hyperbolic', type=bool, default=False, help='Use hyperbolic embeddings')
    parser.add_argument('--num_epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--num_trajectories', type=int, default=10000, help='Number of trajectories')
    parser.add_argument('--maze_type', type=str, default='blank', help='Type of maze')
    args = parser.parse_args()

    if 'blank' in args.maze_type:
        print('blank maze')
        maze = np.zeros((10, 10))
    else:
        maze = create_pyramid(np.zeros((2, 2)), 2)[0]
      # Modify this according to your maze_type logic if needed

    experiment_name = f"experiment_hyperbolic_{args.hyperbolic}_epochs_{args.num_epochs}_trajectories_{args.num_trajectories}_maze_{args.maze_type}"

    wandb.init(
        project="hyperbolic-rl", 
        name=experiment_name, 
        # Track hyperparameters and run metadata
        config={
            "batch_size": 32,
            "embedding_dim": 64,
            "eval_trials": 100,
            "max_steps": 100,
            "hyperbolic": args.hyperbolic,
            "num_epochs": args.num_epochs,
            "temperature": 0.1,
            "num_negatives": 11,
            "learning_rate": 0.001,
            "architecture": "MLP",
            "maze": maze,
            "num_trajectories": args.num_trajectories,
            "maze_type": args.maze_type
        }
    )

    # configs
    config = wandb.config
    print(config)
    manifold = PoincareBall(c=Curvature(value=0.1, requires_grad=True))

    dataset = TrajectoryDataset(maze, config.num_trajectories, embedding_dim=config.embedding_dim, num_negatives=10)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    
    if config.hyperbolic:
        encoder1 = HyperbolicMLP(in_features=4, out_features=config.embedding_dim, manifold=manifold.to(device)).to(device)
        encoder2 = HyperbolicMLP(in_features=2, out_features=config.embedding_dim, manifold=manifold.to(device)).to(device)
        optimizer = RiemannianAdam(list(encoder1.parameters()) + list(encoder2.parameters()), lr=config.learning_rate)
    else:
        encoder1 = StateActionEncoder(config.embedding_dim).to(device)
        encoder2 = StateEncoder(config.embedding_dim).to(device)
        optimizer = optim.Adam(list(encoder1.parameters()) + list(encoder2.parameters()), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.num_epochs):
        total_loss = 0
        for anchor, positive, negatives in dataloader:
            # (s,a) <-> (s)
            anchor = torch.tensor(anchor).to(device, torch.float32)
            positive = torch.tensor(positive).to(device, torch.float32)
            negatives = torch.tensor(negatives).to(device, torch.float32)
            
            if config.hyperbolic:
                m_anchor = manifold_map(anchor, manifold)
                m_positive = manifold_map(positive, manifold)
                m_negatives = manifold_map(negatives, manifold)
            else:
                m_anchor = anchor
                m_positive = positive
                m_negatives = negatives
            
            anchor_enc = encoder1(m_anchor) # takes state, action tuple
            positive_enc = encoder2(m_positive) # takes state
            negatives_enc = encoder2(m_negatives)

            positive_action = anchor[:,[2,3]]
            cur_state = anchor[:,[0,1]]
            angle = torch.arctan2(anchor[:,2], anchor[:,3])

            negative_actions = (angle + torch.pi)[:,None] + (torch.rand(config.num_negatives)[None,:].to(device) - 0.5) * (3 * torch.pi / 2)
            negative_dirs = torch.stack([torch.sin(negative_actions), torch.cos(negative_actions)]).moveaxis(0, -1)
            # print(f'negative actions shape: {negative_actions.shape}')
            # print(negative_dirs.shape)
            negative_full = torch.cat((cur_state.unsqueeze(1).expand(-1, config.num_negatives, -1), negative_dirs), dim=-1).to(device)
            
            if config.hyperbolic:
                m_negative_full = manifold_map(negative_full, manifold)
            else:
                m_negative_full = negative_full

            # print(negative_full.shape)
            neg_action_enc = encoder1(m_negative_full)
            # print(f'positive_enc: {positive_enc.shape}, anchor: {anchor_enc.shape}, neg_action_enc: {neg_action_enc.shape}')
            
            if config.hyperbolic:
                action_loss = hyperbolic_infoNCE_loss(positive_enc, anchor_enc, neg_action_enc, config.temperature, manifold=manifold)
                future_loss = hyperbolic_infoNCE_loss(anchor_enc, positive_enc, negatives_enc, config.temperature, manifold=manifold)
            else:
                action_loss = infoNCE_loss(positive_enc, anchor_enc, neg_action_enc, config.temperature, metric_type=1)
                future_loss = infoNCE_loss(anchor_enc, positive_enc, negatives_enc, config.temperature, metric_type=1)
            
            loss = action_loss + future_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss = total_loss / len(dataloader)
        evals = evaluate(maze, config.eval_trials, encoder1, encoder2, manifold, max_steps=config.max_steps, hyperbolic=config.hyperbolic)
        acc = np.mean([x[2] for x in evals])
        fail = np.mean([x[0] for x in evals])

        metrics = {
            "epoch": epoch + 1,
            "loss": loss,
            "spl": acc,
            "fail": fail
        }
        wandb.log(metrics)

        print(f'Epoch {epoch+1}, Loss: {loss}, SPL: {acc}, Failure %: {fail}')


if __name__ == '__main__':
    main()
