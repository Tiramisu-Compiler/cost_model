import copy
import math
import os
import random
import time
import torch
import wandb
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Calculate the mean absolute percentage error
def mape_criterion(inputs, targets):
    eps = 1e-5
    return 100 * torch.mean(torch.abs(targets - inputs) / (targets + eps))


def train_model(
    config,
    model,
    criterion,
    optimizer,
    max_lr,
    dataloader,
    num_epochs=100,
    log_every=5,
    logger=None,
    train_devices = ["cpu"],
    validation_devices = ["cpu"],
):
    since = time.time()
    losses = []
    train_loss = 0
    best_loss = math.inf
    best_model = None
    hash = random.getrandbits(16)
    dataloader_size = {"train": 0, "val": 0}
    
    # Calculate data size for each dataset. Used to caluclate the loss duriing the training  
    for device in dataloader["train"]:
        for item in dataloader["train"][device]:
            label = item[1]
            dataloader_size["train"] += label.shape[0]
    for device in dataloader["val"]:
        for item in dataloader["val"][device]:
            label = item[1]
            dataloader_size["val"] += label.shape[0]
            
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for phase in ["train", "val"]:
            if phase == "train":
                # Enable gradient tracking for training
                model.train()
            else:
                # Disable gradient tracking for evaluation
                model.eval()

            running_loss = 0.0
            for device in dataloader[phase]:
                pbar = tqdm(dataloader[phase][device])
                # Send model to the training device
                model = model.to(device)
                model.device = device
                
                # Need to recreate the optimizer everytime we change the device we'll be training on because:
                # "In general, you should make sure that optimized parameters live in consistent locations when optimizers are constructed and used."
                # Check the TORCH.OPTIM pytorch documentation for more informations
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=max_lr, weight_decay=0.15e-1
                )
                # Use the 1cycle learning rate policy
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    steps_per_epoch=len(dataloader["train"]),
                    epochs=num_epochs,
                )
                
                # For each batch in the dataset
                for inputs, labels in pbar:
                    # Send the labels and inputs to the training device
                    original_device = labels.device
                    inputs = (
                        inputs[0],
                        inputs[1].to(device),
                        inputs[2].to(device),
                        inputs[3].to(device),
                        inputs[4].to(device),
                        inputs[5].to(device),
                    )
                    labels = labels.to(device)

                    # Reset the gradients for all tensors
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        # Get the model predictions
                        outputs = model(inputs)
                        assert outputs.shape == labels.shape

                        # Calculate the loss
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            # Backpropagation
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                pbar.set_description("Loss: {:.3f}".format(loss.item()))
                running_loss += loss.item() * labels.shape[0]
                # Send the labels back to the original device 
                labels = labels.to(original_device)
                epoch_end = time.time()
            epoch_loss = running_loss / dataloader_size[phase]
            if phase == "val":
                # Append loss to the list of validation losses
                losses.append((train_loss, epoch_loss))
                # If we reached a new minimum loss
                if epoch_loss <= best_loss:
                    best_loss = epoch_loss
                    # Save the model weights at this checkpoint
                    best_model = copy.deepcopy(model)
                    saved_model_path = os.path.join(config.experiment.base_path, "weights/")
                    if not os.path.exists(saved_model_path):
                        os.makedirs(saved_model_path)
                    full_path = os.path.join(
                            saved_model_path,
                            f"best_model_{config.experiment.name}_{hash:4x}.pt",
                        )
                    logger.debug(f"Saving checkpoint to {full_path}")
                    torch.save(
                        model.state_dict(),
                        full_path,
                    )
                # Track progress using the wandb platform
                if config.wandb.use_wandb:
                    wandb.log(
                        {
                            "best_msle": best_loss,
                            "train_msle": train_loss,
                            "val_msle": epoch_loss,
                            "epoch": epoch,
                        }
                    )
                print(
                    "Epoch {}/{}:  "
                    "train Loss: {:.4f}   "
                    "val Loss: {:.4f}   "
                    "time: {:.2f}s   "
                    "best: {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        train_loss,
                        epoch_loss,
                        epoch_end - epoch_start,
                        best_loss,
                    )
                )
                if epoch % log_every == 0:
                    logger.info(
                        "Epoch {}/{}:  "
                        "train Loss: {:.4f}   "
                        "val Loss: {:.4f}   "
                        "time: {:.2f}s   "
                        "best: {:.4f}".format(
                            epoch + 1,
                            num_epochs,
                            train_loss,
                            epoch_loss,
                            epoch_end - epoch_start,
                            best_loss,
                        )
                    )
            else:
                train_loss = epoch_loss
    time_elapsed = time.time() - since

    print(
        "Training complete in {:.0f}m {:.0f}s   "
        "best validation loss: {:.4f}".format(
            time_elapsed // 60, time_elapsed % 60, best_loss
        )
    )
    logger.info(
        "-----> Training complete in {:.0f}m {:.0f}s   "
        "best validation loss: {:.4f}\n ".format(
            time_elapsed // 60, time_elapsed % 60, best_loss
        )
    )
    return losses, best_model
