import time
import argparse
import yaml
import wandb
import torch
import torch.optim as optim
from hypll.optim import RiemannianAdam

import numpy as np
from torch.utils.data import DataLoader

from environments.maze.data import TrajectoryDataset
from networks.losses import explicit_InfoNCE
from utils import save_models, evaluate, get_maze, get_order_function, load_model

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

    experiment_name = f"experiment{config['custom']}_hyperbolic_{config['hyperbolic']}_curvature_{config['curvature']}_learnable_{config['learnable_curvature']}_epochs_{config['num_epochs']}_trajectories_{config['num_trajectories']}_order_{config['order_name']}_maze_{config['maze_type']}_embeddingdim_{config['embedding_dim']}_gamma_{config['gamma']}_batch_{config['batch_size']}"

    # Initialize wandb
    wandb.init(project=config["project"], name=experiment_name, config={**config})
    config = wandb.config
    print(config)

    order_fn = get_order_function(config.order_name)
    maze = get_maze(config["maze_type"])
    dataset = TrajectoryDataset(
        maze,
        config.num_trajectories,
        num_negatives=config.num_negatives,
        gamma=config.gamma,
        order_fn=order_fn,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    print(f"finished dataloading...")

    model_dict = load_model(config, device, pretrained_path='')

    encoder1 = model_dict['encoder1']
    encoder2 = model_dict['encoder2']
    manifold = model_dict['manifold']

    if config.hyperbolic:
        optimizer = RiemannianAdam(
                list(encoder1.parameters()) + list(encoder2.parameters()),
                lr=config['learning_rate'],
            )
    else:
        optimizer = torch.optim.Adam(
                list(encoder1.parameters()) + list(encoder2.parameters()),
                lr=config.learning_rate,
            )
    # Training loop
    total_batches = 0
    start_time = time.time()

    for epoch in range(config.num_epochs):
        total_loss = 0

        acc = 0
        fail = 0

        for anchor, positive, negatives in dataloader:
            # Compute InfoNCE with hard negative future states
            anchor = torch.as_tensor(anchor, dtype=torch.float32, device=device)
            positive = torch.as_tensor(positive, dtype=torch.float32, device=device)
            negatives = torch.as_tensor(negatives, dtype=torch.float32, device=device)

            anchor_enc = encoder1(anchor)  # takes state, action tuple
            positive_enc = encoder2(positive)  # takes state
            negatives_enc = encoder2(negatives)

            future_loss = explicit_InfoNCE(
                anchor_enc,
                positive_enc,
                negatives_enc,
                config.temperature,
                hyperbolic=config.hyperbolic,
                manifold=manifold,
                dot=False,
            )
            # future_loss = infoNCE_loss(anchor_enc, positive_enc, negatives_enc, config.temperature)

            # Compute InfoNCE with hard negative actions
            cur_state = anchor[:, [0, 1]]
            angle = torch.arctan2(anchor[:, 2], anchor[:, 3])
            negative_actions = (angle + torch.pi)[:, None] + (
                torch.rand(config.num_negatives)[None, :].to(device) - 0.5
            ) * (3 * torch.pi / 2)
            negative_dirs = torch.stack(
                [torch.sin(negative_actions), torch.cos(negative_actions)]
            ).moveaxis(0, -1)
            negative_full = torch.cat(
                (
                    cur_state.unsqueeze(1).expand(-1, config.num_negatives, -1),
                    negative_dirs,
                ),
                dim=-1,
            ).to(device)

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
            loss = future_loss + action_loss

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

        if epoch % 32 == 0 and epoch != 0:
            # Runs agent in environment, collects failure and path length metrics
            evals = evaluate(
                maze,
                config.eval_trials,
                encoder1,
                encoder2,
                manifold,
                device,
                max_steps=config.max_steps,
                hyperbolic=config.hyperbolic,
                eps=50.0,
            )
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
