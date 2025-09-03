import math
import time
import torch
import wandb
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm


def mape_criterion(inputs, targets):
    eps = 1e-5
    return 100 * torch.mean(torch.abs(targets - inputs) / (targets + eps))

####################################################
# Main GNN training function
####################################################
def train_gnn_model(
    config,
    model,
    criterion,
    optimizer,
    max_lr,
    dataloader_dict,
    num_epochs,
    log_every,
    logger,
    train_device,
    validation_device,
):
    """
    Trains a PyG-based GNN model using a train/val dataloader.
    
    Args:
      config, logger, etc.: same as before
      model: e.g. your SimpleGCN from modelinggnn.py
      criterion: e.g. mape_criterion
      optimizer: e.g. torch.optim.Adam
      max_lr: for OneCycleLR
      dataloader_dict: {"train": train_loader, "val": val_loader}
        where each loader yields PyG Data objects or merged mini-batches
      num_epochs: ...
      train_device, validation_device: ...
    """
    since = time.time()
    best_loss = math.inf
    best_model = None
    
    dataloader_size = {}
    for phase in ["train", "val"]:
        total_samples = 0
        for batch in dataloader_dict[phase]:
            if isinstance(batch, (list, tuple)):
                data_batch = batch[0]          # (DataBatch, attrs)
            else:
                data_batch = batch              # DataBatch
            total_samples += data_batch.num_graphs
        dataloader_size[phase] = total_samples
    # if initial execution time uncomment this
    # for phase in ["train", "val"]:
    #     total_samples = 0
    #     for data_batch, attr_batch in dataloader_dict[phase]:
    #         print("data batch", data_batch)
    #         total_samples += data_batch.num_graphs  # now works as expected
    #     dataloader_size[phase] = total_samples

    
    # OneCycleLR if you want to step once per mini-batch
    steps_per_epoch = len(dataloader_dict["train"])
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
    )
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                device = train_device
            else:
                model.eval()
                device = validation_device
            
            model = model.to(device)
            model.device = device
            
            running_loss = 0.0
            num_samples_processed = 0
            
            pbar = tqdm(dataloader_dict[phase], desc=f"{phase} Epoch {epoch+1}")
            
            for batch, _ in pbar:
                batch = batch.to(device)
                
                labels = batch.y  # shape [batch_size]
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(batch)  # shape [batch_size]
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                batch_size = labels.shape[0]
                running_loss += loss.item() * batch_size
                num_samples_processed += batch_size
                
                pbar.set_postfix({"loss": loss.item()})
                
                if phase == "train":
                    scheduler.step()
            
            epoch_loss = running_loss / (num_samples_processed + 1e-9)
            
            if phase == "train":
                train_loss = epoch_loss
            else:
                val_loss = epoch_loss
            
            if phase == "val":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model = model.state_dict().copy()
                    saved_model_path = f"{config.experiment.base_path}/weights/"
                    import os
                    if not os.path.exists(saved_model_path):
                        os.makedirs(saved_model_path)
                    model_path = os.path.join(
                        saved_model_path,
                        f"best_gnn_{config.experiment.name}.pt"
                    )
                    torch.save(model.state_dict(), model_path)
                
                if config.wandb.use_wandb:
                    import wandb
                    wandb.log(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "best_loss": best_loss,
                            "epoch": epoch,
                        }
                    )
                
                epoch_time = time.time() - epoch_start_time
                print(
                    f"Epoch {epoch+1}/{num_epochs} => "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Best: {best_loss:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
                if epoch % log_every == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} => "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Best: {best_loss:.4f}, "
                        f"Time: {epoch_time:.2f}s"
                    )
    
    # Done
    total_time = time.time() - since
    print(f"Training complete in {total_time//60:.0f}m {total_time%60:.0f}s, best val loss: {best_loss:.4f}")
    logger.info(f"Training done in {total_time//60:.0f}m {total_time%60:.0f}s, best val loss: {best_loss:.4f}")
    
    return best_loss, best_model
from torch_geometric.data import Batch

def collate_with_attrs(batch):
    data_list, attr_list = zip(*batch)  # separate (Data, attrs) tuples
    return Batch.from_data_list(data_list), attr_list
