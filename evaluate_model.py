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


def speedup_clip(speedup):
    if speedup < 0.01:
        speedup = 0.01
    return speedup


def drop_program(prog_dict):
    if len(prog_dict["schedules_list"]) < 2:
        return True
    if has_skippable_loop_1comp(prog_dict):
        return True
    # drop if we the program is run by lanka24 (because its measurements are inacurate)
    if "node_name" in prog_dict and prog_dict["node_name"] == "lanka24":
        return True
    return False


def drop_schedule(prog_dict, schedule_index):
    schedule_json = prog_dict["schedules_list"][schedule_index]
    schedule_str = sched_json_to_sched_str(schedule_json)
    program_depth = len(prog_dict["program_annotation"]["iterators"])
    # exec time is set to -1 on datapoints that are deemed noisy, or if list empty
    if (not schedule_json["execution_times"]) or min(
        schedule_json["execution_times"]
    ) < 0:
        return True
    # this function works only on single comp programs
    if len(prog_dict["program_annotation"]["computations"]) == 1:
        if sched_is_prunable_1comp(schedule_str, program_depth):
            return True
    if wrongly_pruned_schedule(prog_dict, schedule_index):
        return True

    return False


def default_eval(prog_dict, schedule_index):
    schedule_json = prog_dict["schedules_list"][schedule_index]
    schedule_str = sched_json_to_sched_str(schedule_json)
    program_depth = len(prog_dict["program_annotation"]["iterators"])
    # this function works only on single comp programs
    if len(prog_dict["program_annotation"]["computations"]) == 1:
        return can_set_default_eval_1comp(schedule_str, program_depth)
    else:
        return 0


def fix(map_loc):
    return lambda b: torch.load(io.BytesIO(b), map_location=map_loc)


class MappedUnpickler(pickle.Unpickler):
    def __init__(self, *args, map_location="cpu", **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return fix(self._map_location)
        else:
            return super().find_class(module, name)


def mapped_loads(s, map_location="cpu"):
    bs = io.BytesIO(s)
    unpickler = MappedUnpickler(bs, map_location=map_location)
    return unpickler.load()


def define_model(input_size=776):
    print("Defining the model")
    model = Model_Recursive_LSTM_v2(
        input_size=input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.050] * 5,
        train_device="cuda:0",
        loops_tensor_size=20,
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
            os.path.join(
                "/data/hb2578/cost_model/",
                "factorization_new_dataset/weights/",
                conf.testing.checkpoint,
            ),
            map_location=train_device,
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
