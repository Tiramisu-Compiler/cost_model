import hydra
import gc
import shutil
from hydra.core.config_store import ConfigStore
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *
def generate_dataset(
    base_path,
    batched_data_dir,
    dataset_file_path,
    nb_processes,
    batch_size,
    devices
):
    repr_pkl_output_folder = os.path.join(
        base_path, 
        'pickled/pickled_'
    ) + Path(dataset_file_path).parts[-1][:-4]
    
    # If the pkl files haven't already been generated or if the directory is empty
    if not  os.path.isdir(repr_pkl_output_folder) or not any(os.scandir(repr_pkl_output_folder)):
        print("Loading training data, extracting the model input representation and writing it into pkl files")
        load_data_into_pkls_parallel(
            dataset_file_path,
            nb_processes = nb_processes,
            repr_pkl_output_folder = repr_pkl_output_folder,
            overwrite_existing_pkl = True
        )
    print(f"Reading the pkl files from {repr_pkl_output_folder} into memory for batching")
    dataset, batches_list, indices, device_batches_indices_dict = load_pickled_repr(
        repr_pkl_output_folder,
        max_batch_size = batch_size,
        store_devices = devices
    )
    saved_all_data = False
    device_index = 0
    # Split the training data into multiple parts according to the specified devices
    for device in device_batches_indices_dict:
        starting_index = device_batches_indices_dict[device][0] 
        ending_index = device_batches_indices_dict[device][1]
        device_bl =  batches_list[starting_index:ending_index]
        
        # Shuffling batches to avoid having the same footprint in consecutive batches
        random.shuffle(device_bl)
        
        # Write the batched data into a file
        dataset_path = os.path.join(base_path, batched_data_dir)
        device_extension = "GPU_"+ str(device_index) if "cuda" in str(device) else "CPU"
        file_path = os.path.join(
            base_path, 
            batched_data_dir, 
            f"{Path(dataset_file_path).parts[-1][:-4]}_{device_extension}.pt"
        )
        
        device_index =+ 1
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        
        with open(file_path, "wb") as val_bl_pickle_file:
            torch.save(device_bl, val_bl_pickle_file)
        
    # Delete generated dataset since it has been saved as a file
    del dataset, batches_list, indices
    gc.collect()

@hydra.main(config_path="conf", config_name="config")
def generate_datasets(conf):
    """Converts and split into batches the validation and training dataset.

    Args:
        conf (RecursiveLSTMConfig): The configuration of the repository.
    """
    # Vdlidation
    generate_dataset(
        base_path = conf.experiment.base_path,
        batched_data_dir = "batched/valid",
        dataset_file_path = conf.data_generation.valid_dataset_file,
        nb_processes = conf.data_generation.nb_processes,
        batch_size = conf.data_generation.batch_size,
        devices = conf.training.valid_gpu
    )
    
    # Training
    generate_dataset(
        base_path = conf.experiment.base_path,
        batched_data_dir = "batched/train",
        dataset_file_path = conf.data_generation.train_dataset_file,
        nb_processes = conf.data_generation.nb_processes,
        batch_size = conf.data_generation.batch_size,
        devices = conf.training.train_gpu
    )


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    generate_datasets()
