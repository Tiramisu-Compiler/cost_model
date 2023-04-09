import io

import hydra
import torch
from hydra.core.config_store import ConfigStore

from utils.config import *
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *

def define_and_load_model(conf):
    # Define the model
    model = Model_Recursive_LSTM_v2(
        input_size=conf.model.input_size,
        comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
        drops=list(conf.model.drops),
        loops_tensor_size=8,
        train_device=conf.testing.gpu,
    )
    # Load the trained model weights
    model.load_state_dict(
        torch.load(
            conf.testing.testing_model_weights_path,
            map_location=conf.testing.gpu,
        )
    )
    model = model.to(conf.testing.gpu)
    
    # Set the model to evaluation mode
    model.eval()
    return model


def evaluate(conf, model):
    
    print("Loading the dataset...")
    val_ds, val_bl, val_indices = load_pickled_repr(
        os.path.join(conf.experiment.base_path ,'pickled/pickled_')+Path(conf.data_generation.valid_dataset_file), 
        max_batch_size = 1024, 
        store_device=conf.testing.gpu, 
        train_device=conf.testing.gpu
    )
    print("Evaluation...")
    val_df = get_results_df(val_ds, val_bl, val_indices, model, train_device = conf.testing.gpu)
    val_scores = get_scores(val_df)
    return dict(
        zip(
            ["nDCG", "nDCG@5", "nDCG@1", "Spearman_ranking_correlation", "MAPE"],
            [item for item in val_scores.describe().iloc[1, 1:6].to_numpy()],
        )
    )


@hydra.main(config_path="conf", config_name="config")
def main(conf):
    print("Defining and loading the model using parameters from the config file")
    model = define_and_load_model(conf)
    print(f"Validating on the dataset: {conf.data_generation.valid_dataset_file}")
    scores = evaluate(conf, model)
    print(f"Evaluation scores are:\n{scores}")

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="experiment_config", node=RecursiveLSTMConfig)
    main()
