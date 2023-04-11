import os
import io
import logging
import gc
import hydra
from hydra.core.config_store import ConfigStore
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *
def load_batches_list(
    base_path,
    batched_data_dir,
    dataset_file_path,
    devices,
    device_val='',
):
    batches_list = {}
    index = 0
    
    last_used_device = device_val
    for device in devices:
        storing_device = torch.device(device)
        device_bl = []
        if "cuda" in device:
            extension = "GPU_" + str(index)
            batches_file_path = os.path.join(base_path, batched_data_dir, f"{Path(dataset_file_path).parts[-1][:-4]}_{extension}.pt")
            if os.path.exists(batches_file_path):
                print(f"Loading first part of the set {batches_file_path} into device: {storing_device}")
                with open(batches_file_path, "rb") as file:
                    device_bl = torch.load(batches_file_path, map_location=storing_device)
            batches_list[storing_device] = device_bl
            index =+ 1
            last_used_device = storing_device
    device_bl = []
    batches_file_path = os.path.join( base_path, batched_data_dir, f"{Path(dataset_file_path).parts[-1][:-4]}_CPU.pt")    
    if os.path.exists(batches_file_path):
        print(f"Loading second part of the set {batches_file_path} into the CPU")
        with open(batches_file_path, "rb") as file:
            device_bl = torch.load(batches_file_path, map_location="cpu")
            
        batches_list[last_used_device] = device_bl
    return batches_list

@hydra.main(config_path="conf", config_name="config")
def main(conf):
    # Defining logger
    log_filename = [part for part in conf.training.log_file.split('/') if len(part) > 3][-1]
    log_folder_path = os.path.join(conf.experiment.base_path, "logs/")
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
    log_file = os.path.join(log_folder_path, log_filename)
    logging.basicConfig(filename = log_file,
                        filemode='a',
                        level = logging.DEBUG,
                        format = '%(asctime)s:%(levelname)s:  %(message)s')
    logging.info(f"Starting experiment {conf.experiment.name}")
    
    # Send the model to the first devce we will be training on
    train_device = torch.device(conf.training.train_gpu[0])
    
    # Defining the model
    logging.info("Defining the model")
    model = Model_Recursive_LSTM_v2(
        input_size=conf.model.input_size,
        comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
        drops=list(conf.model.drops),
        loops_tensor_size=8,
        device=conf.training.train_gpu,
    )
    
    # Load model weights and continue training if specified  
    if conf.training.continue_training:
        print(f"Continue training using model from {conf.training.model_weights_path}")
        model.load_state_dict(torch.load(conf.training.model_weights_path, map_location=train_device))
    
    # Enable gradient tracking for training
    for param in model.parameters():
        param.requires_grad = True
        
    # Reading data
    logging.info("Reading the dataset")

    val_bl = load_batches_list(
        base_path = conf.experiment.base_path,
        batched_data_dir = "batched/valid/",
        dataset_file_path = conf.data_generation.valid_dataset_file,
        devices = conf.training.valid_gpu,
        device_val=conf.training.train_gpu[0],
    ) 
    train_bl = load_batches_list(
        base_path = conf.experiment.base_path,
        batched_data_dir = "batched/train/",
        dataset_file_path = conf.data_generation.train_dataset_file,
        devices = conf.training.train_gpu
    ) 
    # Defining training params
    criterion = mape_criterion
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=conf.training.lr, weight_decay=0.15e-1
    )
    logger = logging.getLogger()
    if conf.wandb.use_wandb:
        # Intializing wandb
        wandb.init(project=conf.wandb.project)
        wandb.config = dict(conf)
        wandb.watch(model)
    
    # Training
    print("Training the model")
    bl_dict = {"train": train_bl, "val": val_bl}
    train_model(
        config=conf,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        max_lr=conf.training.lr,
        dataloader=bl_dict,
        num_epochs=conf.training.max_epochs,
        logger=logger,
        log_every=1,
        train_devices=conf.training.train_gpu,
        validation_devices=conf.training.valid_gpu,
    )


if __name__ == "__main__":
    main()
