import os
import io
import logging
    
import hydra
from hydra.core.config_store import ConfigStore

from utils.config import *
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *

@hydra.main(config_path="conf", config_name="config")
def main(config: RecursiveLSTMConfig):
    # Defining logger
    log_filename = [part for part in config.training.log_file.split('/') if len(part) > 3][-1]
    log_folder_path = os.path.join(config.experiment.base_path, "logs/")
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
    log_file = os.path.join(log_folder_path, log_filename)
    logging.basicConfig(filename = log_file,
                        filemode='a',
                        level = logging.DEBUG,
                        format = '%(asctime)s:%(levelname)s:  %(message)s')
    logging.info(f"Starting experiment {config.experiment.name}")
    
    # Reading data
    logging.info("Reading the dataset")
#     bl_dict = read_datasets(config)
    
    # Defining the model
    logging.info("Defining the model")
    model = Model_Recursive_LSTM_v2(
        input_size=config.model.input_size,
        comp_embed_layer_sizes=list(config.model.comp_embed_layer_sizes),
        drops=list(config.model.drops),
        loops_tensor_size=8,
        train_device=config.training.gpu,
    )
    for param in model.parameters():
        param.requires_grad = True
    
    path = os.path.join(
        config.experiment.base_path,
        "dataset_new/train",
        f"{config.data_generation.dataset_name}_1.pt",
    )
    train_device = torch.device(config.training.gpu)
    with open(path, "rb") as file:
        train_bl_1 = torch.load(path, map_location=train_device)
    path = os.path.join(
        config.experiment.base_path,
        "dataset_new/train",
        f"{config.data_generation.dataset_name}_2.pt",
    )
    train_device = torch.device(train_device)
    with open(path, "rb") as file:
        train_bl_2 = torch.load(path, map_location='cpu')
    print("loaded training")
    path = os.path.join(
        config.experiment.base_path,
        "dataset_new/valid",
        f"{config.data_generation.dataset_name}.pt",
    )
    with open(path, "rb") as file:
        val_bl = torch.load(path, map_location='cpu')
        
    # Defining training params
    criterion = mape_criterion
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=0.15e-1
    )
    
    logger = logging.getLogger()
    
    if config.wandb.use_wandb:
        # Intializing wandb
        wandb.init(project=config.wandb.project)
        wandb.config = dict(config)
        wandb.watch(model)

    bl_dict = {"train": train_bl_1+ train_bl_2, "val": val_bl}
    # Training
    
    
    print("Training the model")
    train_model(
        config=config,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        max_lr=config.training.lr,
        dataloader=bl_dict,
        num_epochs=config.training.max_epochs,
        logger=logger,
        log_every=1,
        train_device=config.training.gpu,
    )


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="experiment_config", node=RecursiveLSTMConfig)
    main()
