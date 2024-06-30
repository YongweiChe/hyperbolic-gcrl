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
import os

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
            # if hyperbolic:
            #     goal = manifold_map(goal, manifold=manifold)
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
                    # if hyperbolic:
                    #     cur = manifold_map(cur, manifold)
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


def get_maze(name):
    maze = np.zeros((10, 10))
    
    if 'blank' in name:
        print('blank maze')
        maze = np.zeros((10, 10))
    elif 'slit' in name:
        print('slit maze')
        maze = np.zeros((11, 11))
        maze[:,5] = 1
        maze[5, 5] = 0
    elif 'blocker' in name:
        maze = np.zeros((11, 11))
        maze[3,:] = 1
        maze[3, 10] = 0
    elif 'nested_pyramid' in name:
        maze = create_pyramid(np.zeros((2, 2)), 2)[0]
    else:
        maze = create_pyramid(np.zeros((2, 2)), 1)[0]

    return maze

def save_models(encoder1, encoder2, best_encoder1, best_encoder2, epoch, best_epoch, name=''):
    os.makedirs('models', exist_ok=True)
    torch.save(encoder1.state_dict(), f'models/{name}_encoder1_epoch_{epoch}.pth')
    torch.save(encoder2.state_dict(), f'models/{name}_encoder2_epoch_{epoch}.pth')
    torch.save(best_encoder1, f'models/{name}_best_encoder1_epoch_{best_epoch}.pth')
    torch.save(best_encoder2, f'models/{name}_best_encoder2_epoch_{best_epoch}.pth')

def main():
    parser = argparse.ArgumentParser(description='Maze experiment parameters.')
    parser.add_argument('--project', type=str, default='default', help='Project name')
    parser.add_argument('--custom', type=str, default='1', help='Project name')
    parser.add_argument('--hyperbolic', type=bool, default=False, help='Use hyperbolic embeddings')
    parser.add_argument('--num_epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--num_trajectories', type=int, default=10000, help='Number of trajectories')
    parser.add_argument('--maze_type', type=str, default='blank', help='Type of maze')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of the embeddings')
    parser.add_argument('--gamma', type=float, default=0.1, help='Geometric Distribution factor')
    parser.add_argument('--num_workers', type=int, default=8, help='Num Workers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--max_steps', type=int, default=100, help='max steps')
    parser.add_argument('--hyp_layers', type=int, default=2, help='max steps')

    args = parser.parse_args()

    maze = get_maze(args.maze_type)
      # Modify this according to your maze_type logic if needed

    experiment_name = f"experiment{args.custom}_hyperbolic_{args.hyperbolic}_epochs_{args.num_epochs}_trajectories_{args.num_trajectories}_maze_{args.maze_type}_embeddingdim_{args.embedding_dim}_gamma_{args.gamma}_batch_{args.batch_size}_hyp_layers_{args.hyp_layers}"

    wandb.init(
        project=args.project, 
        name=experiment_name, 
        # Track hyperparameters and run metadata
        config={
            "embedding_dim": args.embedding_dim,
            "eval_trials": 100,
            "max_steps": args.max_steps,
            "hyperbolic": args.hyperbolic,
            "num_epochs": args.num_epochs,
            "temperature": 0.1,
            "batch_size": args.batch_size,
            "num_negatives": args.batch_size,
            "learning_rate": 0.001,
            "architecture": "MLP",
            "maze": maze,
            "num_trajectories": args.num_trajectories,
            "maze_type": args.maze_type,
            "gamma": args.gamma,
            "hyp_layers": args.hyp_layers
        }
    )

    # configs
    config = wandb.config
    print(config)
    manifold = PoincareBall(c=Curvature(value=0.1, requires_grad=True))

    dataset = TrajectoryDataset(maze, config.num_trajectories, embedding_dim=config.embedding_dim, num_negatives=10, gamma=config.gamma)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=args.num_workers)
    
    if config.hyperbolic:
        euc_layers = 4 - config.hyp_layers
        hyp_widths = [64 for _ in range(config.hyp_layers)]
        hyp_widths.append(config.embedding_dim)
        euc_widths = [64 for _ in range(euc_layers)]

        encoder1 = HyperbolicMLP(in_features=4, euc_widths=euc_widths, hyp_widths=hyp_widths, manifold=manifold).to(device)
        encoder2 = HyperbolicMLP(in_features=2, euc_widths=euc_widths, hyp_widths=hyp_widths, manifold=manifold).to(device)
        optimizer = RiemannianAdam(list(encoder1.parameters()) + list(encoder2.parameters()), lr=config.learning_rate)
    else:
        encoder1 = StateActionEncoder(config.embedding_dim).to(device)
        encoder2 = StateEncoder(config.embedding_dim).to(device)
        optimizer = optim.Adam(list(encoder1.parameters()) + list(encoder2.parameters()), lr=config.learning_rate)


    best_spl = 0
    best_encoder1 = encoder1.state_dict()
    best_encoder2 = encoder2.state_dict()
    best_epoch = 0

    # Training loop
    for epoch in range(config.num_epochs):
        total_loss = 0
        for anchor, positive, negatives in dataloader:
            # (s,a) <-> (s)
            anchor = torch.tensor(anchor).to(device, torch.float32)
            positive = torch.tensor(positive).to(device, torch.float32)
            negatives = torch.tensor(negatives).to(device, torch.float32)

            anchor_enc = encoder1(anchor) # takes state, action tuple
            positive_enc = encoder2(positive) # takes state
            negatives_enc = encoder2(negatives)

            cur_state = anchor[:,[0,1]]
            angle = torch.arctan2(anchor[:,2], anchor[:,3])

            negative_actions = (angle + torch.pi)[:,None] + (torch.rand(config.num_negatives)[None,:].to(device) - 0.5) * (3 * torch.pi / 2)
            negative_dirs = torch.stack([torch.sin(negative_actions), torch.cos(negative_actions)]).moveaxis(0, -1)
            # print(f'negative actions shape: {negative_actions.shape}')
            # print(negative_dirs.shape)
            negative_full = torch.cat((cur_state.unsqueeze(1).expand(-1, config.num_negatives, -1), negative_dirs), dim=-1).to(device)
            
            # if config.hyperbolic:
            #     m_negative_full = manifold_map(negative_full, manifold)
            # else:
            #     m_negative_full = negative_full

            # print(negative_full.shape)
            neg_action_enc = encoder1(negative_full)
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
        evals = evaluate(maze, config.eval_trials, encoder1, encoder2, manifold, max_steps=config.max_steps, hyperbolic=config.hyperbolic, eps=50.)
        acc = np.mean([x[2] for x in evals])
        fail = np.mean([x[0] for x in evals])

        metrics = {
            "epoch": epoch + 1,
            "loss": loss,
            "spl": acc,
            "fail": fail
        }
        wandb.log(metrics)

        if acc > best_spl:
            best_spl = acc
            best_encoder1 = encoder1.state_dict()
            best_encoder2 = encoder2.state_dict()
            best_epoch = epoch + 1
    
        if epoch % 32 == 0:
            save_models(encoder1, encoder2, best_encoder1, best_encoder2, epoch + 1, best_epoch, experiment_name)

        print(f'Epoch {epoch+1}, Loss: {loss}, SPL: {acc}, Failure %: {fail}')

    save_models(encoder1, encoder2, best_encoder1, best_encoder2, epoch + 1, best_epoch, experiment_name)

if __name__ == '__main__':
    main()
