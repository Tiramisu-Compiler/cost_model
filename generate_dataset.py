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
#     Validation
    val_ds, val_bl, val_indices= load_data_parallel(conf.data_generation.valid_dataset_file,max_batch_size = 1024,
                                       nb_processes=conf.data_generation.nb_processes,
                                       repr_pkl_output_folder=os.path.join(conf.experiment.base_path ,'pickled/pickled_')+Path(conf.data_generation.valid_dataset_file).parts[-1][:-4],
                                       overwrite_existing_pkl=False, store_device="cpu", train_device="cpu")
    
#     Training
    train_ds, train_bl, train_indices = load_data_parallel(conf.data_generation.train_dataset_file,max_batch_size = 1024,
                                       nb_processes=conf.data_generation.nb_processes,                                       repr_pkl_output_folder = os.path.join(conf.experiment.base_path ,'pickled/pickled_') + Path(conf.data_generation.train_dataset_file).parts[-1][:-4],
                                       overwrite_existing_pkl=False, store_device="cpu", train_device="cpu")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    multiprocessing.set_start_method('spawn')
    cs.store(name="experiment_config", node=RecursiveLSTMConfig)
    generate_datasets()
