import copy
import json
import pickle
import random
import re
import gc
import sys
import multiprocessing
import shutil
import resource
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import enum
import os, psutil
import sympy

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Exceptions
class NbTranformationException(Exception):
    pass

class NbAccessException(Exception):
    pass

class LoopsDepthException(Exception):
    pass

# Global configuration
MAX_NUM_TRANSFORMATIONS = 4
MAX_TAGS = 16
MAX_DEPTH = 5
MAX_EXPR_LEN = 66


#############################################################
#                  GNN DATA BUILDING
#            ( Replaces old LSTM-based approach )
#############################################################
def build_gnn_data_for_schedule(
    function_dict,
    schedule_json,
    device="cpu",
    add_exec_time=False
):
    import torch
    from torch_geometric.data import Data

    # Get the program annotation containing iterators, computations, etc.
    program_annot = function_dict["program_annotation"]
    
    loop_node_features = []
    comp_node_features = []

    loop_name_to_id = {}
    comp_name_to_id = {}
    
    loop_loop_edges = []
    loop_comp_edges = []
    
    # Loop Nodes (each represented as an 8-dim vector)
    i_loop = 0
    for loop_name in program_annot["iterators"]:
        feat = torch.zeros(8, dtype=torch.float) 
        loop_node_features.append(feat)
        loop_name_to_id[loop_name] = i_loop
        i_loop += 1
    
    # Computation Nodes (each represented as a 16-dim vector)
    computations_dict = program_annot["computations"]
    i_comp = 0
    for c_name in sorted(computations_dict.keys(),
                         key=lambda x: computations_dict[x]["absolute_order"]):
        feat = torch.zeros(16, dtype=torch.float)
        if computations_dict[c_name].get("comp_is_reduction", False):
            feat[0] = 1.0
        comp_node_features.append(feat)
        comp_name_to_id[c_name] = i_comp
        i_comp += 1
    
    # Build loop->loop edges from the tree structure
    if "tree_structure" in schedule_json:
        if "roots" in schedule_json["tree_structure"]:
            for root in schedule_json["tree_structure"]["roots"]:
                build_loop_loop_edges(root, None, loop_name_to_id, loop_loop_edges)
    
    # Build loop->computation edges from the tree structure
    if "tree_structure" in schedule_json:
        if "roots" in schedule_json["tree_structure"]:
            for root in schedule_json["tree_structure"]["roots"]:
                build_loop_comp_edges(root, loop_name_to_id, comp_name_to_id, loop_comp_edges)
    
    # Merge node features into a single tensor.
    loop_features = (torch.stack(loop_node_features, dim=0)
                     if loop_node_features else torch.zeros(0, 8))
    comp_features = (torch.stack(comp_node_features, dim=0)
                     if comp_node_features else torch.zeros(0, 16))

    max_dim = max((loop_features.shape[1] if loop_features.shape[0] > 0 else 0),
                  (comp_features.shape[1] if comp_features.shape[0] > 0 else 0))
    if loop_features.shape[1] < max_dim and loop_features.shape[0] > 0:
        pad_cols = max_dim - loop_features.shape[1]
        loop_features = torch.nn.functional.pad(loop_features, (0, pad_cols))
    if comp_features.shape[1] < max_dim and comp_features.shape[0] > 0:
        pad_cols = max_dim - comp_features.shape[1]
        comp_features = torch.nn.functional.pad(comp_features, (0, pad_cols))

    # node features [num_nodes, max_dim]
    x = torch.cat([loop_features, comp_features], dim=0)

    # initial execution time as an extra feature.
    if add_exec_time:
        initial_time_val = float(function_dict["initial_execution_time"])
        exec_time_feat = torch.full((x.shape[0], 1), fill_value=initial_time_val)
        x = torch.cat([x, exec_time_feat], dim=1)

    edge_index_list = []
    for (src, dst) in loop_loop_edges:
        edge_index_list.append((src, dst))
        edge_index_list.append((dst, src))
    
    offset = len(loop_node_features)
    for (l_id, c_id) in loop_comp_edges:
        edge_index_list.append((l_id, offset + c_id))
        edge_index_list.append((offset + c_id, l_id))
    
    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
   
    orig_time = float(function_dict["initial_execution_time"])
    sched_times = schedule_json.get("execution_times", [])
    if sched_times:
        transformed_time = min(sched_times)
    else:
        transformed_time = 1e-9  # avoid division by zero
    final_speedup = orig_time / transformed_time
    y = torch.tensor([final_speedup], dtype=torch.float)

    # Create a PyG Data object.
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )
    return data

def build_loop_loop_edges(node, parent_loop_name, loop_name_to_id, edge_list):
    loop_name = node["loop_name"]
    if parent_loop_name is not None:
        edge_list.append((loop_name_to_id[parent_loop_name], loop_name_to_id[loop_name]))
    for child in node["child_list"]:
        build_loop_loop_edges(child, loop_name, loop_name_to_id, edge_list)

def build_loop_comp_edges(node, loop_name_to_id, comp_name_to_id, edge_list):
    loop_name = node["loop_name"]
    for comp in node["computations_list"]:
        if comp in comp_name_to_id:
            edge_list.append((loop_name_to_id[loop_name], comp_name_to_id[comp]))
    for child in node["child_list"]:
        build_loop_comp_edges(child, loop_name_to_id, comp_name_to_id, edge_list)

def build_loop_feature(loop_name, program_annot, schedule_json):
    """
    Example loop feature. You can replicate your schedule-based logic 
    from get_schedule_representation's 'loop_schedules_dict' step.
    """
    feat = torch.zeros(8, dtype=torch.float)
    return feat

def build_comp_feature(comp_name, program_annot, schedule_json):
    """
    Example computation feature (16-D).
    Could contain comp_is_reduction, transformations, expression embedding, etc.
    """
    feat = torch.zeros(16, dtype=torch.float)
    comp_dict = program_annot["computations"][comp_name]
    if comp_dict["comp_is_reduction"]:
        feat[0] = 1.0
    return feat


#############################################################
#   The "get_func_repr_task_gnn" replaces the old LSTM-based
#   approach. It builds a list of (Data, datapoint_attrs).
#############################################################
# def get_func_repr_task_gnn(input_q, output_q):
#     while not input_q.empty():
#         try:
#             process_id, subset_path, pkl_output_folder, device = input_q.get()

#             # Load the function subset from disk
#             with open(subset_path, 'rb') as f:
#                 programs_dict = pickle.load(f)

#             function_name_list = list(programs_dict.keys())
#             local_list = []

#             for function_name in tqdm(function_name_list, desc=f"Proc-{process_id}"):
#                 func_dict = programs_dict[function_name]

#                 if drop_program(func_dict, function_name):
#                     continue

#                 program_exec_time = func_dict["initial_execution_time"]
#                 data_and_attrs_list = []

#                 for i, sched_json in enumerate(func_dict["schedules_list"]):
#                     if drop_schedule(func_dict, i):
#                         continue

#                     sched_exec_time = np.min(sched_json["execution_times"])
#                     if sched_exec_time <= 0:
#                         continue

#                     speedup_val = program_exec_time / sched_exec_time
#                     speedup_val = speedup_clip(speedup_val)

#                     data_obj = build_gnn_data_for_schedule(
#                         func_dict,
#                         sched_json,
#                         device=device
#                     )

#                     datapoint_attrs = get_datapoint_attributes(function_name, func_dict, i, "<gnn_footprint>")
#                     data_and_attrs_list.append((data_obj, datapoint_attrs))

#                 if len(data_and_attrs_list) > 0:
#                     local_list.append((function_name, data_and_attrs_list))

#             # Save this worker's output to disk
#             pkl_part_filename = os.path.join(pkl_output_folder, f'gnn_representation_part_{process_id}.pkl')
#             with open(pkl_part_filename, 'wb') as f:
#                 pickle.dump(local_list, f, protocol=pickle.HIGHEST_PROTOCOL)

#             output_q.put((process_id, pkl_part_filename))

#         except Exception as e:
#             print(f"[ERROR] Worker {process_id} failed: {e}")


import tempfile

import os
import pickle
import shutil
import json
import tempfile
import multiprocessing
import random
import torch
from pathlib import Path
class GNNDatasetParallel:
    def __init__(
        self,
        dataset_filename=None,
        pkl_output_folder="/home/ih2347/cost_model/outputs/2025-06-04/06-40-26/gnn_pickles",
        nb_processes=4,
        device="cpu",
        just_load_pickled=False
    ):
        self.data_list = []  # list of paths to saved torch files
        self.attr_list = []

        if just_load_pickled:
            print("[INFO] Loading all .pt files directly from folder (ignoring .pkl metadata)...")
            pt_files = sorted(Path(pkl_output_folder).glob("*.pt"))
            if not pt_files:
                print("[WARNING] No .pt files found in folder. Dataset may be empty.")
            for pt_file in pt_files:
                try:
                    _ = torch.load(pt_file)  # Optional: validate loading
                    self.data_list.append(str(pt_file))
                    self.attr_list.append({})  # Placeholder or default attributes
                except Exception as e:
                    print(f"[WARNING] Failed to load {pt_file}: {e}")

        else:
            print("[INFO] Loading all .pt files directly from folder...")
            pt_files = sorted(Path(pkl_output_folder).glob("*.pt"))
            if not pt_files:
                print("[WARNING] No .pt files found in folder. Dataset may be empty.")
            for pt_file in pt_files:
                try:
                    torch.load(pt_file)  # Validate load
                    self.data_list.append(str(pt_file))
                    self.attr_list.append({})  # Dummy attr or adjust as needed
                except Exception as e:
                    print(f"[WARNING] Failed to load {pt_file}: {e}")

            # Optional: generate pickles here if needed in future
            # Not included here for now

        # If you're generating new data (just_load_pickled=False and dataset_filename provided)
        if not just_load_pickled and dataset_filename:
            print("[INFO] Generating new GNN pickled data...")
            if dataset_filename.endswith(".json"):
                with open(dataset_filename, "r") as f:
                    ds_str = f.read()
                programs_dict = json.loads(ds_str)
                del ds_str
            else:
                with open(dataset_filename, "rb") as f:
                    programs_dict = pickle.load(f)

            if os.path.exists(pkl_output_folder):
                shutil.rmtree(pkl_output_folder)
            os.makedirs(pkl_output_folder, exist_ok=True)

            tmp_dir = tempfile.mkdtemp(prefix="subset_", dir=pkl_output_folder)

            manager = multiprocessing.Manager()
            input_q = manager.Queue()
            output_q = manager.Queue()

            fnames = list(programs_dict.keys())
            random.shuffle(fnames)
            chunk_size = (len(fnames) // nb_processes) + 1

            for i in range(nb_processes):
                subset = {k: programs_dict[k] for k in fnames[i * chunk_size: (i + 1) * chunk_size]}
                subset_path = os.path.join(tmp_dir, f"subset_{i}.pkl")
                with open(subset_path, "wb") as f:
                    pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)
                input_q.put((i, subset_path, pkl_output_folder, device))

            processes = []
            for i in range(nb_processes):
                p = multiprocessing.Process(
                    target=get_func_repr_task_gnn,
                    args=(input_q, output_q)
                )
                p.start()
                processes.append(p)

            for _ in range(nb_processes):
                pid, part_file = output_q.get()
                with open(part_file, 'rb') as f:
                    local_list = pickle.load(f)
                for fn, data_attr_pairs in local_list:
                    for (data_path, attrs) in data_attr_pairs:
                        full_data_path = (
                            data_path if os.path.isabs(data_path)
                            else os.path.join(pkl_output_folder, data_path)
                        )
                        if os.path.exists(full_data_path):
                            self.data_list.append(full_data_path)
                            self.attr_list.append(attrs)
                        else:
                            print(f"[WARNING] Missing file: {full_data_path} -- skipping.")

            for p in processes:
                p.join()
            shutil.rmtree(tmp_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_path, attrs = self.data_list[idx], self.attr_list[idx]
        data = torch.load(data_path)
        return data, attrs



def get_func_repr_task_gnn(input_q, output_q):
    while not input_q.empty():
        try:
            process_id, subset_path, pkl_output_folder, device = input_q.get()

            with open(subset_path, 'rb') as f:
                programs_dict = pickle.load(f)

            function_name_list = list(programs_dict.keys())
            local_list = []

            for function_name in tqdm(function_name_list, desc=f"Proc-{process_id}"):
                func_dict = programs_dict[function_name]

                if drop_program(func_dict, function_name):
                    continue

                program_exec_time = func_dict["initial_execution_time"]
                data_and_attrs_list = []

                for i, sched_json in enumerate(func_dict["schedules_list"]):
                    if drop_schedule(func_dict, i):
                        continue

                    sched_exec_time = np.min(sched_json["execution_times"])
                    if sched_exec_time <= 0:
                        continue

                    speedup_val = program_exec_time / sched_exec_time
                    speedup_val = speedup_clip(speedup_val)

                    data_obj = build_gnn_data_for_schedule(
                        func_dict,
                        sched_json,
                        device=device
                    )

                    datapoint_attrs = get_datapoint_attributes(function_name, func_dict, i, "<gnn_footprint>")

                    # Save data_obj to individual file
                    data_filename = f"{function_name}_{i}.pt"
                    full_data_path = os.path.join(pkl_output_folder, data_filename)
                    torch.save(data_obj, full_data_path)

                    data_and_attrs_list.append((data_filename, datapoint_attrs))  # only save filename


                if len(data_and_attrs_list) > 0:
                    local_list.append((function_name, data_and_attrs_list))

            pkl_part_filename = os.path.join(pkl_output_folder, f'gnn_representation_part_{process_id}.pkl')
            with open(pkl_part_filename, 'wb') as f:
                pickle.dump(local_list, f, protocol=pickle.HIGHEST_PROTOCOL)

            output_q.put((process_id, pkl_part_filename))

        except Exception as e:
            print(f"[ERROR] Worker {process_id} failed: {e}")



def drop_program(prog_dict, prog_name):
    if len(prog_dict["schedules_list"]) < 2:
        return True
    return False

def drop_schedule(prog_dict, schedule_index):
    schedule_json = prog_dict["schedules_list"][schedule_index]
    if (not schedule_json["execution_times"]) or min(schedule_json["execution_times"]) < 0:
        return True
    return False

def speedup_clip(speedup):
    if speedup < 0.01:
        speedup = 0.01
    return speedup

def get_datapoint_attributes(func_name, program_dict, schedule_index, tree_footprint):
    schedule_json = program_dict["schedules_list"][schedule_index]
    sched_id = str(schedule_index).zfill(4)
    sched_str = get_schedule_str(program_dict["program_annotation"], schedule_json)
    exec_time = np.min(schedule_json["execution_times"])
    memory_use = program_dict["program_annotation"]["memory_size"]
    node_name = program_dict["node_name"] if "node_name" in program_dict else "unknown"
    speedup = program_dict["initial_execution_time"] / exec_time
    return (
        func_name,
        sched_id,
        sched_str,
        exec_time,
        memory_use,
        node_name,
        tree_footprint,
        speedup,
    )


def get_schedule_str(program_json, sched_json):
    comp_name = [
        n
        for n in sched_json.keys()
        if not n in ["unfuse_iterators", "tree_structure", "execution_times", "fusions", "sched_str", "legality_check", "exploration_method"]
    ]
    sched_str = ""
    if ("fusions" in sched_json and sched_json["fusions"]):
        for fusion in sched_json["fusions"]:
            sched_str += "F("
            for name in comp_name:
                if name in fusion:
                    sched_str += name + ","
            sched_str = sched_str[:-1]
            sched_str += ")"
    for name in comp_name:
        transf_loop_nest = program_json["computations"][name]["iterators"].copy()
        schedule = sched_json[name]
        if ("fusions" in sched_json and sched_json["fusions"]):
            for fusion in sched_json["fusions"]:
                if name in fusion:
                    iterator_comp_name = fusion[0]
                    transf_loop_nest = program_json["computations"][iterator_comp_name]["iterators"].copy()
                    schedule = sched_json[iterator_comp_name]
        sched_str += '{' + name + '}:'
        for transformation in schedule["transformations_list"]:
            if (transformation[0] == 1):
                sched_str += "I(L" + str(transformation[1]) + ",L" + str(transformation[2]) + ")"
                assert(transformation[1]<len(transf_loop_nest) and transformation[2]<len(transf_loop_nest))
                tmp_it = transf_loop_nest[transformation[1]]
                transf_loop_nest[transformation[1]] = transf_loop_nest[transformation[2]]
                transf_loop_nest[transformation[2]] = tmp_it
            elif (transformation[0] == 2):
                sched_str += "R(L" + str(transformation[3])+ ")"
            elif (transformation[0] == 3):
                sched_str += "S(L" + str(transformation[4]) + ",L" + str(transformation[5]) + "," + str(transformation[6]) + "," + str(transformation[7]) + ")"
        if schedule["parallelized_dim"]:
            dim_index = transf_loop_nest.index(schedule["parallelized_dim"])
            sched_str += "P(L" + str(dim_index) + ")"
        if schedule["shiftings"]:
            for shifting in schedule['shiftings']:
                dim_index = transf_loop_nest.index(shifting[0])
                sched_str += "Sh(L" + str(dim_index) + "," + str(shifting[1])+")"
        if schedule["tiling"]:
            # omitted for brevity, same code as before
            pass
        if schedule["unrolling_factor"]:
            dim_index = len(transf_loop_nest) - 1
            dim_name = transf_loop_nest[-1]
            sched_str += "U(L" + str(dim_index) + "," + schedule["unrolling_factor"] + ")"
            transf_loop_nest[dim_index : dim_index + 1] = (
                dim_name + "_Uouter",
                dim_name + "_Uinner",
            )
    return sched_str



if __name__ == "__main__":
    # If you want to create GNN Data pickles from a dataset:
    dataset_path = "/home/ih2347/cost_model/data_samples/LOOPer_dataset_train.pkl"  # or .pkl
    pkl_folder = "/home/ih2347/cost_model/gnn_pickles"
    
    # Build the dataset in parallel
    gnn_dataset = GNNDatasetParallel(
        dataset_filename=dataset_path,
        pkl_output_folder=pkl_folder,
        nb_processes=4,
        device="cuda:0",
        just_load_pickled=False  
    )
    
    print("Number of GNN graphs:", len(gnn_dataset))
    
    loader = DataLoader(gnn_dataset.data_list, batch_size=8, shuffle=True)
    for batch in loader:
        print(batch.x.shape, batch.edge_index.shape, batch.y.shape)
