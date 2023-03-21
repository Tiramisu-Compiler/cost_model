import io

import hydra
import torch
from hydra.core.config_store import ConfigStore

from utils.config import *
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *

train_device = torch.device("cuda")
store_device = torch.device("cuda")

def define_model():
    model = Model_Recursive_LSTM_v2(
        input_size=conf.model.input_size,
        comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
        drops=list(conf.model.drops),
        loops_tensor_size=8,
        train_device=conf.training.gpu,
    ).to(train_device)
    return model


def evaluate(model, dataset_path):
    print("Loading the dataset...")
    batch = torch.load(dataset_path)
    val_ds, val_bl, val_indices = batch
    print("Evaluation...")
    val_df = get_results_df(val_ds, val_bl, val_indices, model, train_device="cpu")
    val_scores = get_scores(val_df)
    return dict(
        zip(
            ["nDCG", "nDCG@5", "nDCG@1", "Spearman_ranking_correlation", "MAPE"],
            [item for item in val_scores.describe().iloc[1, 1:6].to_numpy()],
        )
    )


@hydra.main(config_path="conf", config_name="config")
def main(conf):
    model = define_model(input_size=776)
    model.load_state_dict(
        torch.load(
            conf.testing.model_weights_path,
            map_location=conf.training.gpu,
        )
    )
    for dataset in conf.testing.datasets:
        if dataset in ["valid", "bench"]:
            print(f"getting results for {dataset}")
            dataset_path = os.path.join(
                conf.experiment.base_path,
                f"dataset/{dataset}",
                f"{conf.data_generation.dataset_name}.pt",
            )
            scores = evaluate(model, dataset_path)
            print(scores)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="experiment_config", node=RecursiveLSTMConfig)
    main()
