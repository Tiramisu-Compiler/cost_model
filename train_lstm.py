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
    
    train_ds, train_bl, train_indices= load_pickled_repr(repr_pkl_output_folder= os.path.join(config.experiment.base_path ,'pickled/pickled_')+Path(config.data_generation.train_dataset_file).parts[-1][:-4],
                                          max_batch_size = 1024, store_device=config.training.gpu, train_device=config.training.gpu)
    del train_ds.programs_dict
    del train_ds.X
    val_ds, val_bl, val_indices= load_pickled_repr(repr_pkl_output_folder=os.path.join(config.experiment.base_path ,'pickled/pickled_')+Path(config.data_generation.valid_dataset_file).parts[-1][:-4],
                                          max_batch_size = 1024, store_device=config.training.gpu, train_device=config.training.gpu)
    
    del val_ds.programs_dict
    del val_ds.X
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

    bl_dict={'train':train_bl, 'val':val_bl}
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
