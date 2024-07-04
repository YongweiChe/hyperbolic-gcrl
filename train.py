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
from continuous_maze import bfs, gen_traj, plot_traj, ContinuousGridEnvironment, TrajectoryDataset, LabelDataset
from hyperbolic_networks import HyperbolicMLP, hyperbolic_infoNCE_loss, manifold_map
from networks import StateActionEncoder, StateEncoder, infoNCE_loss
import os
import time

import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(maze, num_trials, encoder1, encoder2, manifold, max_steps=100, hyperbolic=False, eps=10., step_size=0.5, verbose=False):
    """
    Run policy on maze, collect failure metrics
    """
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
    """
    Pre-set mazes
    """
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
    elif 'island' in name:
        maze = np.zeros((11, 11))
        maze[3:8, 5:7] = 1
    else:
        maze = create_pyramid(np.zeros((2, 2)), 1)[0]

    return maze

def get_order_function(name):
    """
    order trajectories by some arbitrary metric. (e.g.) horizontal is where trajectories can only end to the right of the start
    """
    def order_by_second_coordinate(point1, point2):
        return point1[1] - point2[1]
    
    def order_by_first_coordinate(point1, point2):
        return point1[0] - point2[0]
    
    def order_by_distance_from_origin(point1, point2):
        dist1 = math.sqrt(point1[0]**2 + point1[1]**2)
        dist2 = math.sqrt(point2[0]**2 + point2[1]**2)
        return dist1 - dist2
    
    name = name.lower()
    if 'horizontal' in name:
        print(f'horizontal order fn')
        return order_by_second_coordinate
    elif 'vertical' in name:
        print(f'vertical order fn')
        return order_by_first_coordinate
    elif 'origin' in name:
        print(f'dist order fn')
        return order_by_distance_from_origin
    else:
        print(f'no order')
        return None


def save_models(encoder1, encoder2,epoch, name=''):
    os.makedirs('models', exist_ok=True)
    torch.save(encoder1.state_dict(), f'models/{name}_encoder1_epoch_{epoch}.pth')
    torch.save(encoder2.state_dict(), f'models/{name}_encoder2_epoch_{epoch}.pth')

@profile
def main():
    parser = argparse.ArgumentParser(description='Run experiment with config file.')
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()

    # Load the config file
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    maze = get_maze(config['maze_type'])
    experiment_name = f"experiment{config['custom']}_hyperbolic_{config['hyperbolic']}_epochs_{config['num_epochs']}_trajectories_{config['num_trajectories']}_order_{config['order_name']}_maze_{config['maze_type']}_embeddingdim_{config['embedding_dim']}_gamma_{config['gamma']}_batch_{config['batch_size']}_hyp_layers_{config['hyp_layers']}"

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
    print(config)
    manifold = PoincareBall(c=Curvature(value=0.1, requires_grad=True))
    order_fn = get_order_function(config.order_name)
    dataset = TrajectoryDataset(maze, 
                                config.num_trajectories, 
                                embedding_dim=config.embedding_dim, 
                                num_negatives=config.num_negatives, 
                                gamma=config.gamma,
                                order_fn=order_fn
                                )
    
    for i in range(10): # make sure order is functioning correctly :)
        print(f'{dataset[i][0][:2]} -> {dataset[i][1]}')

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    if config.hyperbolic:
        # weird math for experimenting with number of hyperbolic layers. Need to make more modular
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


    # Training loop
    total_batches = 0
    start_time = time.time()
    
    print('starting...')
    for epoch in range(config.num_epochs):
        total_loss = 0

        acc = 0
        fail = 0

        for anchor, positive, negatives in dataloader:
            # (s,a) <-> (s)
            anchor = torch.as_tensor(anchor, dtype=torch.float32, device=device)
            positive = torch.as_tensor(positive, dtype=torch.float32, device=device)
            negatives = torch.as_tensor(negatives, dtype=torch.float32, device=device)

            anchor_enc = encoder1(anchor) # takes state, action tuple
            positive_enc = encoder2(positive) # takes state
            negatives_enc = encoder2(negatives)

            cur_state = anchor[:,[0,1]]
            angle = torch.arctan2(anchor[:,2], anchor[:,3])

            # Symmetric, InfoNCE with actions as the negative now, (s, a) <-> (a)
            negative_actions = (angle + torch.pi)[:,None] + (torch.rand(config.num_negatives)[None,:].to(device) - 0.5) * (3 * torch.pi / 2)
            negative_dirs = torch.stack([torch.sin(negative_actions), torch.cos(negative_actions)]).moveaxis(0, -1)
            negative_full = torch.cat((cur_state.unsqueeze(1).expand(-1, config.num_negatives, -1), negative_dirs), dim=-1).to(device)

            neg_action_enc = encoder1(negative_full)

            if config.hyperbolic:
                action_loss = hyperbolic_infoNCE_loss(positive_enc, anchor_enc, neg_action_enc, config.temperature, manifold=manifold)
                future_loss = hyperbolic_infoNCE_loss(anchor_enc, positive_enc, negatives_enc, config.temperature, manifold=manifold)
            else:
                action_loss = infoNCE_loss(positive_enc, anchor_enc, neg_action_enc, config.temperature, metric_type=1)
                future_loss = infoNCE_loss(anchor_enc, positive_enc, negatives_enc, config.temperature, metric_type=1)

            loss = future_loss + action_loss
            
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

        if epoch % 32 == 0 and epoch != 0:
            # Runs agent in environment, collects failure and path length metrics
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

            print(f'Epoch {epoch+1}, Loss: {loss}, SPL: {acc}, Failure %: {fail}')

            save_models(encoder1, encoder2, epoch + 1, experiment_name)
        else:
            metrics = {
                    "epoch": epoch + 1,
                    "loss": loss
                }
            wandb.log(metrics)  
            print(f'Epoch {epoch+1}, Loss: {loss}')

    save_models(encoder1, encoder2, epoch + 1, experiment_name)

if __name__ == '__main__':
    main()
