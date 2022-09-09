import os
import io

import hydra
from hydra.core.config_store import ConfigStore

from utils.config import *
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *


def read_datasets(config):
    path = os.path.join(
        config.experiment.base_path,
        "dataset/valid",
        f"{config.data_generation.dataset_name}.pt",
    )
    with open(path, "rb") as file:
        valid_ds, valid_bl, valid_indices = torch.load(path)
        del valid_ds, valid_indices

    path = os.path.join(
        config.experiment.base_path,
        "dataset/train",
        f"{config.data_generation.dataset_name}.pt",
    )
    with open(path, "rb") as file:
        train_ds, train_bl, train_indices = torch.load(path)
        del train_ds, train_indices

    bl_dict = {"train": train_bl, "val": valid_bl}
    return bl_dict


@hydra.main(config_path="conf", config_name="config")
def main(config: RecursiveLSTMConfig):
    global lstm_embed
    # Reading data

    bl_dict = read_datasets(config)
    train_device = "cuda:0"

    model = Model_Recursive_LSTM_v2(
        input_size=config.model.input_size,
        comp_embed_layer_sizes=list(config.model.comp_embed_layer_sizes),
        drops=list(config.model.drops),
        loops_tensor_size=20,
        train_device=train_device,
    )
    for param in model.parameters():
        param.requires_grad = True

    # Defining training params
    criterion = mape_criterion
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=0.15e-1
    )
    log_file = os.path.join(config.experiment.base_path, config.training.log_file)

    if config.wandb.use_wandb:
        # Intializing wandb
        wandb.init(project=config.wandb.project)
        wandb.config = dict(config)
        wandb.watch(model)

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
        log_file=log_file,
        log_every=1,
        train_device=train_device,
    )


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="experiment_config", node=RecursiveLSTMConfig)
    main()
