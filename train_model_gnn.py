import os
import io
import logging
import random
import gc
import hydra
from hydra.core.config_store import ConfigStore

import torch
from torch_geometric.loader import DataLoader

from utils_gnn.train_utils import train_gnn_model, mape_criterion
from utils_gnn.data_utils import GNNDatasetParallel  
from utils_gnn.modeling import SimpleGCN, SimpleGAT, ResidualGIN
from torch_geometric.data import Batch

def collate_with_attrs(batch):
    """
    Custom collate function for batching GNN data and accompanying attributes.
    """
    data_list, attr_list = zip(*batch)
    batched_data = Batch.from_data_list(data_list)
    return batched_data, list(attr_list)
@hydra.main(config_path="conf", config_name="config-gnn")
def main(conf):
    log_folder_path = os.path.join(conf.experiment.base_path, "logs/")
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
    log_file = os.path.join(log_folder_path, "gnn_training.log")
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s: %(message)s'
    )
    logger = logging.getLogger()

    # Setup wandb
    if conf.wandb.use_wandb:
        import wandb
        wandb.init(name=f"{conf.experiment.name}_{conf.model.name}", project=conf.wandb.project)
        wandb.config = dict(conf)
    
    # Decide on device
    train_device = torch.device(conf.training.training_gpu)
    val_device = torch.device(conf.training.validation_gpu)

    # --- 1) Load or create your GNN dataset ---
    gnn_dataset_train = GNNDatasetParallel(
        dataset_filename=conf.data_generation.train_dataset_file,
        pkl_output_folder="gnn_pickles/train",
        nb_processes=4,
        device="cuda:1",
        just_load_pickled=True
    )

    gnn_dataset_val = GNNDatasetParallel(
        dataset_filename=conf.data_generation.valid_dataset_file,
        pkl_output_folder="gnn_pickles/val",
        nb_processes=4,
        device="cuda:1",
        just_load_pickled=True
    )


    # --- 2) Make PyG DataLoaders ---
    train_loader = DataLoader(
        gnn_dataset_train, 
        batch_size=conf.data_generation.batch_size, 
        shuffle=True,
        collate_fn=collate_with_attrs,
    )
    val_loader = DataLoader(
        gnn_dataset_val,
        batch_size=conf.data_generation.batch_size,
        shuffle=False,
        collate_fn=collate_with_attrs,
    )
    dataloaders = {"train": train_loader, "val": val_loader}

    model = ResidualGIN(
        in_channels=conf.model.input_size,       
        hidden_channels=conf.model.hidden_size,  
        out_channels=1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf.training.lr,
        weight_decay=1e-2
    )
    
    best_loss, best_model_state = train_gnn_model(
        config=conf,
        model=model,
        criterion=mape_criterion,
        optimizer=optimizer,
        max_lr=conf.training.lr,
        dataloader_dict=dataloaders,
        num_epochs=conf.training.max_epochs,
        logger=logger,
        log_every=1,
        train_device=train_device,
        validation_device=val_device,
    )
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

if __name__ == "__main__":
    main()
