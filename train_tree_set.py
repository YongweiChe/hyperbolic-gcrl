import time
import argparse
import yaml
import wandb
import torch
import torch.optim as optim
from hypll.optim import RiemannianAdam
from torch.optim.lr_scheduler import StepLR

import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from environments.maze.data import set_collate_fn
from environments.tree.data import SetDataset
from environments.tree.tree import NaryTreeEnvironment
from networks.losses import symmetrized_InfoNCE
from utils import save_models, load_tree_model

import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="Run experiment with config file.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    # Load the config file
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    experiment_name = f"set_experiment{config['custom']}_hyperbolic_{config['hyperbolic']}_curvature_{config['curvature']}_epochs_{config['num_epochs']}_trajectories_{config['num_trajectories']}_order_{config['order_name']}_depth_{config['depth']}_branch_{config['branching_factor']}_embeddingdim_{config['embedding_dim']}_gamma_{config['gamma']}_batch_{config['batch_size']}"

    # Initialize wandb
    wandb.init(project=config["project"], name=experiment_name, config={**config})
    config = wandb.config
    print(config)

    dataset = SetDataset(
        depth=config.depth,
        branching_factor=config.branching_factor,
        num_trajectories=config.num_trajectories,
        num_negatives=config.num_negatives,
        gamma=config.gamma
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=set_collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    tree = NaryTreeEnvironment(depth=config.depth, branching_factor=config.branching_factor)
    num_states = tree.num_nodes
    num_actions = tree.branching_factor + 2
    model_dict = load_tree_model(config, num_states, num_actions, device, pretrained_path='')

    encoder1 = model_dict['encoder1']
    encoder2 = model_dict['encoder2']
    manifold = model_dict['manifold']

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

    # Add the scheduler here
    scheduler = StepLR(optimizer, step_size=config.lr_scheduler_step_size, gamma=config.lr_scheduler_gamma)

    for name, param in encoder2.named_parameters():
        print(name)
    
    print(encoder2)
    # Training loop
    total_batches = 0
    start_time = time.time()

    def check_nan_inf(model):
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                print(f"NaN or Inf detected in {name}")

    encoder2.train()  # Set model to training mode

    for epoch in range(config.num_epochs):
        total_loss = 0
        total_batches = 0

        for batch in dataloader:
            s1 = torch.as_tensor(batch['set1'], dtype=torch.int64, device=device)
            s2 = torch.as_tensor(batch['set2'], dtype=torch.int64, device=device)

            mask1 = torch.as_tensor(batch['set1_mask'], dtype=torch.bool, device=device)
            mask2 = torch.as_tensor(batch['set2_mask'], dtype=torch.bool, device=device)

            s1_enc = encoder2(s1, mask1)
            s2_enc = encoder2(s2, mask2)

            try:
                loss = symmetrized_InfoNCE(s1_enc, s2_enc, config.temperature, device, hyperbolic=config.hyperbolic, manifold=manifold)

                optimizer.zero_grad()
                loss.backward()

                # Check for NaN or Inf gradients
                for name, param in encoder2.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN or Inf gradient detected in {name}")
                            # You might want to skip this batch or take other corrective action

                # Gradient clipping
                clip_grad_norm_(encoder2.parameters(), max_norm=1.0)
                check_nan_inf(encoder2)
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1

            except RuntimeError as e:
                print(f"Error in batch: {e}")
                continue

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

        metrics = {"epoch": epoch + 1, "loss": loss, "lr": scheduler.get_last_lr()[0]}
        wandb.log(metrics)
        print(f"Epoch {epoch+1}, Loss: {loss}, LR: {scheduler.get_last_lr()[0]}")

        if epoch % 32 == 0 and epoch != 0:
            save_models(config, encoder1, encoder2, epoch, experiment_name)

        scheduler.step()  # Step the scheduler at the end of each epoch

    save_models(config, encoder1, encoder2, epoch + 1, experiment_name)


if __name__ == "__main__":
    main()