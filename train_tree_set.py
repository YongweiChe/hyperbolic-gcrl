import time
import argparse
import yaml
import wandb
import torch
import torch.optim as optim
from hypll.optim import RiemannianAdam

import numpy as np
from torch.utils.data import DataLoader

from environments.maze.data import set_collate_fn
from environments.tree.data import SetDataset
from networks.losses import symmetrized_InfoNCE
from utils import save_models, evaluate, get_maze, get_order_function, load_model, load_tree_model

import wandb
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

    # load model
    # config.architecture = 'DeepSet'
    # print(f'architecture: {config.architecture}')
    model_dict = load_tree_model(config, device, pretrained_path='')

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
                list(encoder2.parameters()),
                lr=config.learning_rate,
            )


    for name, param in encoder2.named_parameters():
        print(name)
    
    print(encoder2)
    # Training loop
    total_batches = 0
    start_time = time.time()

    for epoch in range(config.num_epochs):
        total_loss = 0

        for batch in dataloader:
            s1 = torch.as_tensor(batch['set1'], dtype=torch.int64, device=device)
            s2 = torch.as_tensor(batch['set2'], dtype=torch.int64, device=device)

            mask1 = torch.as_tensor(batch['set1_mask'], dtype=torch.int64, device=device)
            mask2 = torch.as_tensor(batch['set2_mask'], dtype=torch.int64, device=device)
            # print(f's1 shape: {s1.shape}, s2 shape: {s2.shape}')
            # print(f's1 shape: {s1.shape}, mask shape: {mask1.shape}')
            # print(mask1)
            s1_enc = encoder2(s1, mask1)
            s2_enc = encoder2(s2, mask2)

            loss = symmetrized_InfoNCE(s1_enc, s2_enc, config.temperature, device, hyperbolic=config.hyperbolic, manifold=manifold)

            optimizer.zero_grad()
            loss.backward()

            # for name, param in encoder2.named_parameters():
            #     print(f"{name}: {param.grad}")

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
            save_models(config, encoder1, encoder2, epoch, experiment_name)
    save_models(config, encoder1, encoder2, epoch + 1, experiment_name)


if __name__ == "__main__":
    main()
