import time
import argparse
import yaml
import wandb
import torch
import torch.optim as optim
from hypll.optim import RiemannianAdam

import numpy as np
from torch.utils.data import DataLoader

from environments.tree.data import GeneralTrajectoryDataset, get_tree_trajectories, get_maze_trajectories
from environments.tree.tree import NaryTreeEnvironment
from environments.tree.discrete_maze import GridMazeEnvironment
from networks.losses import explicit_InfoNCE
from utils import save_models, evaluate, get_maze, get_order_function, load_tree_model, eval_tree, eval_maze

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "--hyperbolic",
        type=lambda x: (str(x).lower() == 'true'),
        help="Toggle hyperbolic setting (true/false)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Set the depth as an integer",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        help="Set the embedding dimension as an integer",
    )
    args = parser.parse_args()

    # Load the config file
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    # Override config with command line arguments if provided
    if args.hyperbolic is not None:
        config['hyperbolic'] = args.hyperbolic
    if args.depth is not None:
        config['depth'] = args.depth
    if args.embedding_dim is not None:
        config['embedding_dim'] = args.embedding_dim

    maze_type = ''
    if config['maze_type']:
        maze_type = config['maze_type']
    
    experiment_name = f"env_{config['env']}_experiment{config['custom']}__maze_type_{maze_type}_hyperbolic_{config['hyperbolic']}_symmetric_{config['symmetric']}_curvature_{config['curvature']}_learnable_{config['learnable_curvature']}_epochs_{config['num_epochs']}_trajectories_{config['num_trajectories']}_depth_{config['depth']}_branch_{config['branching_factor']}_embeddingdim_{config['embedding_dim']}_gamma_{config['gamma']}_batch_{config['batch_size']}"

    # Initialize wandb
    wandb.init(project=config["project"], name=experiment_name, config={**config})
    config = wandb.config
    print(config)

    tree = NaryTreeEnvironment(depth=config.depth, branching_factor=config.branching_factor)
    trajectories = get_tree_trajectories(tree, config.num_trajectories)

    if config.env == 'maze':
        raw_maze = get_maze(config.maze_type)
        maze = GridMazeEnvironment(raw_maze)
        print(f'maze:\n{maze}')
        trajectories = get_maze_trajectories(maze, config.num_trajectories)
        num_states = maze.num_states
        num_actions = 5
        dataset = GeneralTrajectoryDataset(
            trajectories=trajectories,
            valid_indices=maze.get_unflattened_valid_indices(),
            num_negatives=config.num_negatives,
            gamma=config.gamma
        )
    elif config.env == 'tree':
        tree = NaryTreeEnvironment(depth=config.depth, branching_factor=config.branching_factor)
        trajectories = get_tree_trajectories(tree, config.num_trajectories)
        num_states = tree.num_nodes
        num_actions = tree.branching_factor + 2
        dataset = GeneralTrajectoryDataset(
            trajectories=trajectories,
            valid_indices=[i for i in range(tree.num_nodes)],
            num_negatives=config.num_negatives,
            gamma=config.gamma
        )
    else:
        raise Exception("Select a valid discrete environment in the config.")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    print(f"finished dataloading...")

    model_dict = load_tree_model(config, num_states, num_actions, device, pretrained_path='')

    encoder1 = model_dict['encoder1']
    encoder2 = model_dict['encoder2']
    manifold = model_dict['manifold']

    print(f'encoder1: {encoder1}')
    print(f'encoder2: {encoder2}')

    if config.hyperbolic:
        optimizer = RiemannianAdam(
                list(encoder1.parameters()) + list(encoder2.parameters()),
                lr=config.learning_rate,
            )
    else:
        optimizer = torch.optim.Adam(
                list(encoder1.parameters()) + list(encoder2.parameters()),
                lr=config.learning_rate,
            )
    # Training loop
    total_batches = 0
    start_time = time.time()

    print(f'symmetric: {config.symmetric}')
    for epoch in range(config.num_epochs):
        total_loss = 0

        acc = 0
        fail = 0

        for anchor, positive, negatives in dataloader:
            # Compute InfoNCE with hard negative future states
            anchor = torch.as_tensor(anchor, device=device)
            positive = torch.as_tensor(positive, device=device)
            negatives = torch.as_tensor(negatives, device=device)
            if config.symmetric:
                # print(f'anchor.shape: {anchor[:,0][:,None].shape}')
                anchor_enc = encoder2(anchor[:,0][:,None]) # takes state
            else:
                anchor_enc = encoder1(anchor)  # takes state, action tuple
            positive_enc = encoder2(positive)  # takes state
            negatives_enc = encoder2(negatives)

            # print(f'anchor_enc: {anchor_enc.shape}')
            future_loss = explicit_InfoNCE(
                anchor_enc,
                positive_enc,
                negatives_enc,
                config.temperature,
                hyperbolic=config.hyperbolic,
                manifold=manifold,
                dot=False,
            )

            loss = future_loss
            # future_loss = infoNCE_loss(anchor_enc, positive_enc, negatives_enc, config.temperature)

            # Compute InfoNCE with hard negative actions

            def generate_negative_samples(state, action, num_actions):
                batch_size = state.shape[0]
                device = state.device

                # Create a range tensor for each item in the batch
                range_tensor = torch.arange(num_actions, device=device).unsqueeze(0).expand(batch_size, -1)

                # Create a mask to exclude the actual action for each item in the batch
                mask = range_tensor != action.unsqueeze(1)

                # Use the mask to select negative actions
                neg_actions = range_tensor[mask].view(batch_size, -1)

                # Repeat state for each negative action
                repeated_states = state.unsqueeze(1).expand(-1, num_actions - 1)

                # Combine repeated states with negative actions
                negative_full = torch.stack([repeated_states, neg_actions], dim=-1)

                return negative_full


            if not config.symmetric:
                state = anchor[:,0]
                action = anchor[:,1]
                
                negative_full = generate_negative_samples(state, action, num_actions)
                neg_action_enc = encoder1(negative_full)

                action_loss = explicit_InfoNCE(
                    positive_enc,
                    anchor_enc,
                    neg_action_enc,
                    config.temperature,
                    hyperbolic=config.hyperbolic,
                    manifold=manifold,
                    dot=False,
                )
                # action_loss = infoNCE_loss(positive_enc, anchor_enc, neg_action_enc, config.temperature)
                # backprop
                loss += action_loss

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

        if epoch % 8 == 0 and epoch != 0:
            # Runs agent in environment, collects failure and path length metrics
            if config.env == 'tree':
                evals = eval_tree(
                    config.depth,
                    config.branching_factor,
                    config.eval_trials,
                    encoder1,
                    encoder2,
                    manifold,
                    device,
                    max_steps=(4 * config.depth), # give some room for error
                    hyperbolic=config.hyperbolic,
                )
            elif config.env == 'maze':
                evals = eval_maze(
                    raw_maze,
                    config.eval_trials,
                    encoder1,
                    encoder2,
                    manifold,
                    device,
                    max_steps=(4 * config.depth), # give some room for error
                    hyperbolic=config.hyperbolic,
                )
                acc = np.mean([x[2] for x in evals])
                fail = np.mean([x[0] for x in evals])
            else:
                evals = [(0, 0, 0)]

            acc = np.mean([x[2] for x in evals])
            fail = np.mean([x[0] for x in evals])

            metrics = {"epoch": epoch + 1, "loss": loss, "spl": acc, "fail": fail}
            wandb.log(metrics)

            print(f"Epoch {epoch+1}, Loss: {loss}, SPL: {acc}, Failure %: {fail}")

            save_models(config, encoder1, encoder2, epoch, experiment_name)
        else:
            metrics = {"epoch": epoch + 1, "loss": loss}
            wandb.log(metrics)
            print(f"Epoch {epoch+1}, Loss: {loss}")

    save_models(config, encoder1, encoder2, epoch + 1, experiment_name)


if __name__ == "__main__":
    main()
