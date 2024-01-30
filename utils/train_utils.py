import copy
import math
import os
import random
import time
import torch
import wandb
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
# Calculate the mean absolute percentage error
def mape_criterion(inputs, targets):
    eps = 1e-5
    return 100 * torch.mean(torch.abs(targets - inputs) / (targets + eps))

def total_loss(inputs, targets):

    y_pred = inputs
    y_true = targets
    indices = torch.tensor(list(range(10))).to(y_pred.device)
    y_pred_sliced = torch.index_select(y_pred, 1, indices)
    y_true_sliced = torch.index_select(y_true, 1, indices)
    return torch.stack([ce_loss(y_pred_sliced, y_true_sliced), bce_loss(y_pred[:,10], y_true[:,10]), bce_loss(y_pred[:,11], y_true[:,11]), bce_loss(y_pred[:,12], y_true[:,12]), bce_loss(y_pred[:,13], y_true[:,13])], dim=0).sum(dim=0)/5

def i_loss(inputs, targets):
    return

def ce_loss(inputs, targets):
    ce_loss = nn.CrossEntropyLoss()
    output = ce_loss(inputs, torch.max(targets,1)[1])
    return output

def bce_loss(inputs, targets):
    bce = nn.BCELoss()
    loss = bce(inputs, targets)
    return loss

def BinaryCrossEntropy_i(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

def train_model(
    config,
    model,
    criterion,
    optimizer,
    max_lr,
    dataloader,
    num_epochs,
    log_every,
    logger,
    train_device,
    validation_device,
):

    since = time.time()
    losses = []
    train_loss = 0
    best_loss = math.inf
    best_model = None
    hash = random.getrandbits(16)
    dataloader_size = {"train": 0, "val": 0}
    # Calculate data size for each dataset. Used to caluclate the loss duriing the training  
    for item in dataloader["train"]:
        label = item[1]
        dataloader_size["train"] += label.shape[0]
    for item in dataloader["val"]:
        label = item[1]
        dataloader_size["val"] += label.shape[0]
    
    # Use the 1cycle learning rate policy
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(dataloader["train"]),
        epochs=num_epochs,
    )
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for phase in ["train", "val"]:
            if phase == "train":
                # Enable gradient tracking for training
                model.train()
                device = train_device
            else:
                # Disable gradient tracking for evaluation
                model.eval()
                # If the user wants to run the validation on another GPU
                if (validation_device != "cpu"):
                    device = validation_device
            # Send model to the training device
            model = model.to(device)
            model.device = device
            
            device
            running_loss = 0.0
            pbar = tqdm(dataloader[phase])
            
            # For each batch in the dataset
            for inputs, labels in pbar:
                # Send the labels and inputs to the training device
                original_device = labels.device
                inputs = (
                    inputs[0],
                    inputs[1].to(device),
                    inputs[2].to(device),
                    inputs[3].to(device),
                )
                labels = labels.to(device)
                
                # Reset the gradients for all tensors
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # Get the model predictions
                    outputs = model(inputs)
                    # print(f"labels shape {labels.shape} outputs shape {outputs.shape}")
                    assert outputs.shape == labels.shape
                    
                    # Calculate the loss
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        # Backpropagation
                        loss.backward()
                        optimizer.step()
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
                scheduler.step()
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
