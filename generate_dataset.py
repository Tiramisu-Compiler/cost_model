import hydra
from hydra.core.config_store import ConfigStore

from utils.config import *
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *


@hydra.main(config_path="conf", config_name="config")
def generate_datasets(conf):
    """Converts and split into batches the validation and training dataset.

    Args:
        conf (RecursiveLSTMConfig): The configuration of the repository.
    """
    # Load pre trained model
    model = Model_Recursive_LSTM_v2(
                input_size=conf.model.input_size,
                comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
                drops=list(conf.model.drops),
                loops_tensor_size=8,
                train_device=conf.training.gpu,
            )
    # Load the trained weights
    model.load_state_dict(torch.load(conf.training.model_weights_path, map_location=torch.device(conf.training.gpu)))
    model.eval()
    # Validation
#     val_ds, val_bl, val_indices, gpu_fitted_batches_index = load_data_parallel(
#         conf.data_generation.valid_dataset_file, 
#         max_batch_size = 1024, 
#         model=model, 
#         nb_processes=conf.data_generation.nb_processes, 
#         repr_pkl_output_folder= os.path.join(
#             conf.experiment.base_path ,'pickled/pickled_'
#             )+Path(conf.data_generation.valid_dataset_file).parts[-1][:-4],
#         overwrite_existing_pkl=True, 
#         store_device=conf.training.gpu, 
#         train_device=conf.training.gpu)
    
#     validation_dataset_path = os.path.join(conf.experiment.base_path,"dataset_new/valid")
    
#     if not os.path.exists(validation_dataset_path):
#         os.makedirs(validation_dataset_path)
#     with open(
#         os.path.join(
#             conf.experiment.base_path,
#             "dataset_new/valid/",
#             f"{conf.data_generation.dataset_name}.pt",
#         ),
#         "wb",
#     ) as valid_bl_pickle_file:
#         torch.save(val_bl, valid_bl_pickle_file)
    
#     Training
    train_ds, train_bl, train_indices, gpu_fitted_batches_index = load_data_parallel(
        conf.data_generation.train_dataset_file,
        max_batch_size = 1024,
        model=model,
        nb_processes=conf.data_generation.nb_processes, 
        repr_pkl_output_folder = os.path.join(
            conf.experiment.base_path ,'pickled/pickled_'
        ) + Path(conf.data_generation.train_dataset_file).parts[-1][:-4],
        overwrite_existing_pkl=True, 
        store_device=conf.training.gpu, 
        train_device=conf.training.gpu)
    
    training_dataset_path = os.path.join(conf.experiment.base_path, "dataset_new/train")
    
    train_bl_1 = train_bl[:gpu_fitted_batches_index]
    train_bl_2 = train_bl[gpu_fitted_batches_index:]
    
    if not os.path.exists(training_dataset_path):
        os.makedirs(training_dataset_path)
    with open(
        os.path.join(
            conf.experiment.base_path,
            "dataset_new/train/",
            f"{conf.data_generation.dataset_name}_1.pt",
        ),
        "wb",
    ) as train_bl_pickle_file:
        torch.save(train_bl_1, train_bl_pickle_file)
    with open(
        os.path.join(
            conf.experiment.base_path,
            "dataset_new/train/",
            f"{conf.data_generation.dataset_name}_2.pt",
        ),
        "wb",
    ) as train_bl_pickle_file:
        torch.save(train_bl_2, train_bl_pickle_file)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    multiprocessing.set_start_method('spawn')
    cs.store(name="experiment_config", node=RecursiveLSTMConfig)
    generate_datasets()
