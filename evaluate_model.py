import io

import hydra
import torch
from hydra.core.config_store import ConfigStore

from utils.config import *
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *

def define_and_load_model(config):
    model = Model_Recursive_LSTM_v2(
        input_size=config.model.input_size,
        comp_embed_layer_sizes=list(config.model.comp_embed_layer_sizes),
        drops=list(config.model.drops),
        loops_tensor_size=8,
        train_device=config.testing.gpu,
    )
    model.load_state_dict(
        torch.load(
            config.testing.testing_model_weights_path,
            map_location=config.testing.gpu,
        )
    )
    model = model.to(config.testing.gpu)
    model.eval()
    return model


def evaluate(model, dataset_path, train_device):
    print("Loading the dataset...")
    with open(dataset_path, "rb") as file:
        validation_data = torch.load(dataset_path, map_location='cpu')
    val_ds, val_bl, val_indices = validation_data
    print("Evaluation...")
    val_df = get_results_df(val_ds, val_bl, val_indices, model, train_device = train_device)
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
    path = os.path.join(
        conf.experiment.base_path,
        "dataset_new/valid",
        f"{conf.data_generation.dataset_name}.pt",
    )
    print(f"Validating on the dataset: {path}")
    scores = evaluate(model, path, conf.testing.gpu)
    print(f"Evaluation scores are:\n{scores}")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="experiment_config", node=RecursiveLSTMConfig)
    main()
