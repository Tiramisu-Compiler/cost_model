import hydra
import gc
import shutil
from hydra.core.config_store import ConfigStore
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *


@hydra.main(config_path="conf", config_name="config")
def generate_datasets(conf):
    """Converts and split into batches the validation and training dataset.

    Args:
        conf (RecursiveLSTMConfig): The configuration of the repository.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = [
        conf.training.training_gpu,
        conf.training.validation_gpu
    ] if conf.training.validation_gpu != "cpu" else conf.training.training_gpu
    # Validation
    val_repr_pkl_output_folder = os.path.join(
        conf.experiment.base_path, 
        'pickled/pickled_'
    ) + Path(conf.data_generation.valid_dataset_file).parts[-1][:-4]
    
    # If the pkl files haven't already been generated or if the directory is empty
    if not  os.path.isdir(val_repr_pkl_output_folder) or not any(os.scandir(val_repr_pkl_output_folder)):
        print("Loading training data, extracting the model input representation and writing it into pkl files")
        load_data_into_pkls_parallel(
            conf.data_generation.valid_dataset_file,
            nb_processes = conf.data_generation.nb_processes,
            repr_pkl_output_folder = val_repr_pkl_output_folder,
            overwrite_existing_pkl = True
        )
    print(f"Reading the pkl files from {val_repr_pkl_output_folder} into memory for batching")
    val_ds, val_bl, val_indices, gpu_fitted_batches_index = load_pickled_repr(
        val_repr_pkl_output_folder,
        max_batch_size = conf.data_generation.batch_size,
        store_device = "cpu",
        train_device = "cpu"
    )
    
    # Split the training data into two parts such that the first part fits directly into the GPU
    val_bl_1 = val_bl[:gpu_fitted_batches_index]
    val_bl_2 = val_bl[gpu_fitted_batches_index:]
    
    # Shuffling batches to avoid having the same footprint in consecutive batches
    random.shuffle(val_bl_1)
    random.shuffle(val_bl_2)
    
    # Write the first part of the batched validation data into a file
    validation_dataset_path = os.path.join(conf.experiment.base_path, "batched/valid")
    valid_file_path = os.path.join(
        conf.experiment.base_path, 
        "batched/valid/", 
        f"{Path(conf.data_generation.valid_dataset_file).parts[-1][:-4]}_GPU.pt"
    )
    
    if not os.path.exists(validation_dataset_path):
        os.makedirs(validation_dataset_path)
        
    with open(valid_file_path, "wb") as val_bl_pickle_file:
        torch.save(val_bl_1, val_bl_pickle_file)
        
    # Write the second part of the batched validation data into a file
    if len(val_bl_2) > 0:
        # We test if the second part is not empty becuase there are cases where all the data fits in the GPU
        valid_file_path = os.path.join( 
            conf.experiment.base_path, 
            "batched/valid/", 
            f"{Path(conf.data_generation.valid_dataset_file).parts[-1][:-4]}_CPU.pt"
        )

        if not os.path.exists(validation_dataset_path):
            os.makedirs(validation_dataset_path)

        with open(valid_file_path, "wb") as val_bl_pickle_file:
            torch.save(val_bl_2, val_bl_pickle_file)
        
    # Delete generated dataset since it has been saved as a file
    del val_ds, val_bl, val_indices
    gc.collect()
    
#     #Training
    train_repr_pkl_output_folder = os.path.join(
        conf.experiment.base_path,
        'pickled/pickled_'
    ) + Path(conf.data_generation.train_dataset_file).parts[-1][:-4]
    
    # If the pkl files haven't already been generated
    if not  os.path.isdir(train_repr_pkl_output_folder) or not any(os.scandir(train_repr_pkl_output_folder)):
        print("Loading training data, extracting the model input representation and writing it into pkl files")
        load_data_into_pkls_parallel(
            conf.data_generation.train_dataset_file,
            nb_processes = conf.data_generation.nb_processes,
            repr_pkl_output_folder = train_repr_pkl_output_folder,
            overwrite_existing_pkl = True
        )
    print(f"Reading the pkl files from {train_repr_pkl_output_folder} into memory for batching")
    train_ds, train_bl, train_indices, gpu_fitted_batches_index = load_pickled_repr(
        train_repr_pkl_output_folder,
        max_batch_size = conf.data_generation.batch_size,
        store_device = "cuda:0",
        train_device = "cuda:0"
    )
    
    
    # Split the training data into two parts such that the first part fits directly into the GPU
    train_bl_1 = train_bl[:gpu_fitted_batches_index]
    train_bl_2 = train_bl[gpu_fitted_batches_index:]
    
    # Shuffling batches to avoid having the same footprint in consecutive batches
    random.shuffle(train_bl_1)
    random.shuffle(train_bl_2)
    
    # Write the first part of the batched training data into a file
    train_dataset_path = os.path.join(conf.experiment.base_path, "batched/train")
    train_file_path = os.path.join(
        conf.experiment.base_path, 
        "batched/train/", 
        f"{Path(conf.data_generation.train_dataset_file).parts[-1][:-4]}_GPU.pt"
    )
    
    if not os.path.exists(train_dataset_path):
        os.makedirs(train_dataset_path)
        
    with open(train_file_path, "wb") as train_bl_pickle_file:
        torch.save(train_bl_1, train_bl_pickle_file)
        
    # Write the second part of the batched training data into a file
    if len(train_bl_2) > 0:
        # We test if the second part is not empty becuase there are cases where all the data fits in the GPU
        train_file_path = os.path.join( 
            conf.experiment.base_path, 
            "batched/train/", 
            f"{Path(conf.data_generation.train_dataset_file).parts[-1][:-4]}_CPU.pt"
        )

        if not os.path.exists(train_dataset_path):
            os.makedirs(train_dataset_path)

        with open(train_file_path, "wb") as train_bl_pickle_file:
            torch.save(train_bl_2, train_bl_pickle_file)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    generate_datasets()
