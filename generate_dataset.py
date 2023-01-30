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
                                       overwrite_existing_pkl=True, store_device=conf.training.gpu, train_device=conf.training.gpu)
    
#     Training
    train_ds, train_bl, train_indices = load_data_parallel(conf.data_generation.train_dataset_file,max_batch_size = 1024,
                                       nb_processes=conf.data_generation.nb_processes,                                       repr_pkl_output_folder = os.path.join(conf.experiment.base_path ,'pickled/pickled_') + Path(conf.data_generation.train_dataset_file).parts[-1][:-4],
                                       overwrite_existing_pkl=True, store_device=conf.training.gpu, train_device=conf.training.gpu)
    
#     dataset_path = os.path.join(conf.experiment.base_path, "dataset/")
#     if not os.path.exists(dataset_path):
#         os.makedirs(dataset_path)

# #     # Validation
# #     valid_ds, _, _, valid_bl, valid_indices = load_data(
# #         conf.data_generation.valid_dataset_file,
# #         split_ratio=0,
# #         max_batch_size=conf.data_generation.batch_size,
# #         drop_sched_func=drop_schedule,
# #         drop_prog_func=drop_program,
# #         default_eval=can_set_default_eval,
# #         speedups_clip_func=speedup_clip,
# #     )
# #     validation_dataset_path = os.path.join(conf.experiment.base_path, "dataset/valid")
# #     if not os.path.exists(validation_dataset_path):
# #         os.makedirs(validation_dataset_path)
# #     with open(
# #         os.path.join(
# #             conf.experiment.base_path,
# #             "dataset/valid/",
# #             f"{conf.data_generation.dataset_name}.pt",
# #         ),
# #         "wb",
# #     ) as valid_bl_pickle_file:
# #         torch.save([valid_ds, valid_bl, valid_indices], valid_bl_pickle_file)
        
# #     # Training
# #     train_ds, _, _, train_bl, train_indices = load_data(
# #         conf.data_generation.train_dataset_file,
# #         split_ratio=0,
# #         max_batch_size=conf.data_generation.batch_size,
# #         drop_sched_func=drop_schedule,
# #         drop_prog_func=drop_program,
# #         default_eval=can_set_default_eval,
# #         speedups_clip_func=speedup_clip,
# #     )

# #     training_dataset_path = os.path.join(conf.experiment.base_path, "dataset/train")
# #     if not os.path.exists(training_dataset_path):
# #         os.makedirs(training_dataset_path)
# #     with open(
# #         os.path.join(
# #             conf.experiment.base_path,
# #             "dataset/train/",
# #             f"{conf.data_generation.dataset_name}.pt",
# #         ),
# #         "wb",
# #     ) as train_bl_pickle_file:
# #         torch.save([train_ds, train_bl, train_indices], train_bl_pickle_file)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="experiment_config", node=RecursiveLSTMConfig)
    generate_datasets()
