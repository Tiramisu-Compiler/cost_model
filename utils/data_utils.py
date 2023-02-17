import copy
import json
import pickle
import random
import re
import gc
import multiprocessing
import shutil
import resource
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils.config import *
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import enum
# An exception to limit the maximum number of allowed transformations 
class NbTranformationException(Exception):
    pass

class RandomMatrix(Exception):
    pass

# An exception to limit the maximum number of read-write accesses. 
class NbAccessException(Exception):
    pass

# An exception to limit the maximum number of nested loops. Currently set to 5.
class LoopsDepthException(Exception):
    pass

# Maximum sequence of transformations (reversal, interchange and skewing) allowed. Currently set to 4 
MAX_NUM_TRANSFORMATIONS = 4

# Maximum size of the tags vector representing each transformation
MAX_TAGS = 8

# Enumeration for the different exploration algorithms used to generate the data
# class Exploration_method(int, enum.Enum):
#    Beam_search = 0
#    Recursive_beam_search = 1
#    Reinforcement_learning = 2

# Creates a template for the input representation
def get_representation_template(program_dict, max_depth, train_device="cpu"):
    # Set the max and min number of accesses allowed 
    max_accesses = 15
    min_accesses = 0

    comps_repr_templates_list = []
    comps_expr_repr_templates_list = []
    comps_indices_dict = dict()
    comps_placeholders_indices_dict = dict()

    # Get the program JSON represenation
    program_json = program_dict["program_annotation"]
    
    # Get the computations (program statements) dictionary and order them according to the absolute_order attribute
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )

    for comp_index, comp_name in enumerate(ordered_comp_list):
        
        comp_dict = computations_dict[comp_name]
        expr_dict = comp_dict["expression_representation"]
        comp_type = comp_dict["data_type"]
        comps_expr_repr_templates_list.append(get_tree_expr_repr(expr_dict, comp_type))
        if len(comp_dict["accesses"]) > max_accesses:
            raise NbAccessException
        
        if len(comp_dict["accesses"]) < min_accesses:
            raise NbAccessException
        
        if len(comp_dict["iterators"]) > max_depth:
            raise LoopsDepthException

        comp_repr_template = []

        comp_repr_template.append(+comp_dict["comp_is_reduction"])

        iterators_repr = []
        # Add a representation of each loop of this computation
        
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
            # TODOF does this work when iterators have the same name?
            iterator_dict = program_json["iterators"][iterator_name]
            # Create a unique code for each loop
            c_code = "C" + str(comp_index)
            l_code = c_code + "-L" + str(iter_i)
            
            # Add a placeholder for transformations applied to this loop
            iterators_repr.extend(
                [
                    l_code + "Parallelized",
                    l_code + "Tiled",
                    l_code + "TileFactor",
                    l_code + "Fused",
                    l_code + "Shifted",
                    l_code + "ShiftFactor",
                ]
            )
        
        iterator_repr_size = int(len(iterators_repr) / len(comp_dict["iterators"]))
        
        # Add padding incase the number of loops is lower than the max
        iterators_repr.extend(
            [0] * iterator_repr_size * (max_depth - len(comp_dict["iterators"]))
        )
        
        # Add two tags for whether unrolling was applied and the unrolling factor
        iterators_repr.extend([c_code + "-Unrolled", c_code + "-UnrollFactor"])

        # Add a placeholder for the other transformations to be applied (skewing, reversal and interchage)
        iterators_repr.append(c_code + "-TransformationTagsStart")
        iterators_repr.extend(["M"] * (MAX_TAGS * MAX_NUM_TRANSFORMATIONS - 2))
        iterators_repr.append(c_code + "-TransformationTagsEnd")
        
        # Adding initial constraint matrix
        # Remove the 1 mask from constraint matrix. Not necessary.
        iterators_repr.append(c_code+'-OgConstraintMatrixStart')
        iterators_repr.extend(['OgC']*((max_depth*max_depth*2)-2))
        iterators_repr.append(c_code+'-OgConstraintMatrixEnd')
        
        # Adding initial constraint vector
        iterators_repr.append(c_code+'-OgConstraintVectorStart')
        iterators_repr.extend(['V']*(max_depth*2-2))
        iterators_repr.append(c_code+'-OgConstraintVectorEnd')
        
        # Adding transformed constraint matrix
        iterators_repr.append(c_code+'-ConstraintMatrixStart')
        iterators_repr.extend(['C']*((max_depth*max_depth*2)-2))
        iterators_repr.append(c_code+'-ConstraintMatrixEnd')
                              
        # Add the loop representation to the computation vector 
        comp_repr_template.extend(iterators_repr)
        
        # Pad the write access matrix and add it to the representation
        padded_write_matrix = pad_access_matrix(
            isl_to_write_matrix(comp_dict["write_access_relation"]), max_depth
        )
        write_access_repr = [
            comp_dict["write_buffer_id"] + 1
        ] + padded_write_matrix.flatten().tolist()
        comp_repr_template.extend(write_access_repr)

        # Pad the read access matrix and add it to the representation
        # Todo add details about the read accesses 
        read_accesses_repr = []
        for read_access_dict in comp_dict["accesses"]:
            read_access_matrix = pad_access_matrix(
                read_access_dict["access_matrix"], max_depth
            )
            read_access_repr = (
                [+read_access_dict["access_is_reduction"]]
                + [read_access_dict["buffer_id"] + 1]
                + read_access_matrix.flatten().tolist()
            )
            read_accesses_repr.extend(read_access_repr)

        
        access_repr_len = (max_depth + 1) * (max_depth + 2) + 1 + 1
        
        read_accesses_repr.extend(
            [0] * access_repr_len * (max_accesses - len(comp_dict["accesses"]))
        )
        comp_repr_template.extend(read_accesses_repr)
        
        comps_repr_templates_list.append(comp_repr_template)
        # Create a mapping between the features and their position in the representation
        comps_indices_dict[comp_name] = comp_index
        for j, element in enumerate(comp_repr_template):
            if isinstance(element, str):
                comps_placeholders_indices_dict[element] = (comp_index, j)
            
        
    # Create a representation of the loops independantly from the computations
    loops_repr_templates_list = []
    loops_indices_dict = dict()
    loops_placeholders_indices_dict = dict()

    for loop_index, loop_name in enumerate(program_json["iterators"]):
        # Create a unique code for each loop
        loop_repr_template = []
        l_code = "L" + loop_name
        
        # Add a placeholder for transformations applied to this loop
        loop_repr_template.extend(
            [
                l_code + "Parallelized",
                l_code + "Tiled",
                l_code + "TileFactor",
                l_code + "Fused",
                l_code + "Unrolled",
                l_code + "UnrollFactor",
                l_code + "Shifted",
                l_code + "ShiftFactor",
            ]
        )
        
        # Create a mapping between the features and their position in the representation
        loops_repr_templates_list.append(loop_repr_template)
        loops_indices_dict[loop_name] = loop_index

        for j, element in enumerate(loop_repr_template):
            if isinstance(element, str):
                loops_placeholders_indices_dict[element] = (loop_index, j)
    
    # Get the original version of the program 
    no_sched_json = program_dict["schedules_list"][0]
    
    # Make sure no fusion was applied on this version and get the original tree structure 
    assert "fusions" not in no_sched_json or no_sched_json["fusions"] == None
    
    orig_tree_structure = no_sched_json["tree_structure"]
    tree_annotation = copy.deepcopy(orig_tree_structure)
    
   
    prog_tree = update_tree_atributes(tree_annotation, loops_indices_dict, comps_indices_dict, train_device=train_device)
    
    return (
        prog_tree,
        comps_repr_templates_list,
        loops_repr_templates_list,
        comps_placeholders_indices_dict,
        loops_placeholders_indices_dict,
        comps_expr_repr_templates_list,
    )

# TODO add description
def update_tree_atributes(node, loops_indices_dict, comps_indices_dict, train_device="cpu"):
        if "roots" in node :
            for root in node["roots"]:
                update_tree_atributes(root, loops_indices_dict, comps_indices_dict, train_device=train_device)
            return node

        node["loop_index"] = torch.tensor(loops_indices_dict[node["loop_name"]]).to(
            train_device
        )
        if node["computations_list"] != []:
            node["computations_indices"] = torch.tensor(
                [
                    comps_indices_dict[comp_name]
                    for comp_name in node["computations_list"]
                ]
            ).to(train_device)
            node["has_comps"] = True
        else:
            node["has_comps"] = False
        for child_node in node["child_list"]:
            update_tree_atributes(child_node, loops_indices_dict, comps_indices_dict, train_device=train_device)
        return node

    
    
# Fill the representation template with the corresponsing schedule features 
def get_schedule_representation(
    program_json,
    schedule_json,
    comps_repr_templates_list,
    loops_repr_templates_list,
    comps_placeholders_indices_dict,
    loops_placeholders_indices_dict,
    max_depth,
):
    
    # Create a copy of the templates to avoid modifying the values for other schedules
    comps_repr = copy.deepcopy(comps_repr_templates_list)
    loops_repr = copy.deepcopy(loops_repr_templates_list)

    
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )
    
    
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = program_json["computations"][comp_name]
        comp_schedule_dict = schedule_json[comp_name]
    
        fused_levels = []
        
        # If fusion was applied, save which two loops were fused together
        if "fusions" in schedule_json and schedule_json["fusions"]:
            for fusion in schedule_json["fusions"]:

                if comp_name in fusion:
                    fused_levels.append(fusion[2])

        
        c_code = "C" + str(comp_index)
        # Loop representation for this computation
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):

            l_code = c_code + "-L" + str(iter_i)
            
            # Check whether parallelization was applied and put the tag in its corresponding position in the computation representation
            parallelized = 0
            if iterator_name == comp_schedule_dict["parallelized_dim"]:
                parallelized = 1
            p_index = comps_placeholders_indices_dict[l_code + "Parallelized"]
            comps_repr[p_index[0]][p_index[1]] = parallelized
            
            # Check whether tiling was applied and put the tags in their corresponding position in the computation representation
            tiled = 0
            tile_factor = 0
            if comp_schedule_dict["tiling"] and (
                iterator_name in comp_schedule_dict["tiling"]["tiling_dims"]
            ):
                tiled = 1
                tile_factor_index = comp_schedule_dict["tiling"]["tiling_dims"].index(
                    iterator_name
                )
                tile_factor = int(
                    comp_schedule_dict["tiling"]["tiling_factors"][tile_factor_index]
                )
            p_index = comps_placeholders_indices_dict[l_code + "Tiled"]
            comps_repr[p_index[0]][p_index[1]] = tiled
            p_index = comps_placeholders_indices_dict[l_code + "TileFactor"]
            comps_repr[p_index[0]][p_index[1]] = tile_factor

            # Check whether fusion was applied and put the tag in its corresponding position in the computation representation
            fused = 0
            if iter_i in fused_levels:
                fused = 1
            p_index = comps_placeholders_indices_dict[l_code + "Fused"]
            comps_repr[p_index[0]][p_index[1]] = fused
            
            shifted = 0
            shifting_factor = 0
            if comp_schedule_dict['shiftings']:
                for shifting in comp_schedule_dict['shiftings']: 
                    if iterator_name.startswith(shifting[0]): # loof if the current loop is being shifted
                        shifted=1
                        shifting_factor = shifting[1]
                        break
            p_index = comps_placeholders_indices_dict[l_code + "Shifted"]
            comps_repr[p_index[0]][p_index[1]] = shifted
            p_index = comps_placeholders_indices_dict[l_code + "ShiftFactor"]
            comps_repr[p_index[0]][p_index[1]] = shifting_factor
        # Check whether unrolling was applied and put the tags in their corresponding position in the computation representation
        unrolled = 0
        unroll_factor = 0
        if comp_schedule_dict["unrolling_factor"]:
            unrolled = 1
            unroll_factor = int(comp_schedule_dict["unrolling_factor"])
            
        p_index = comps_placeholders_indices_dict[c_code + "-Unrolled"]
        comps_repr[p_index[0]][p_index[1]] = unrolled
        
        p_index = comps_placeholders_indices_dict[c_code + "-UnrollFactor"]
        comps_repr[p_index[0]][p_index[1]] = unroll_factor
        
        # Check which transformations (interchange, reversal and skweing) were applied and add the padded vector representation to their corresponding position
        padded_tags = get_padded_transformation_tags(
            program_json, schedule_json, comp_name, max_depth
        )
        
        tags_start = comps_placeholders_indices_dict[ c_code + "-TransformationTagsStart" ]
        
        tags_end = comps_placeholders_indices_dict[c_code + "-TransformationTagsEnd"]
        
        nb_tags_elements = tags_end[1] - tags_start[1] + 1
        
        assert len(padded_tags) == nb_tags_elements
        
        comps_repr[tags_start[0]][tags_start[1] : tags_end[1] + 1] = padded_tags
        
        ogc_start = comps_placeholders_indices_dict[c_code+'-OgConstraintMatrixStart']
        
        ogc_end = comps_placeholders_indices_dict[c_code+'-OgConstraintMatrixEnd']
        
        nb_mat_elements = ogc_end[1] - ogc_start[1] + 1
        
        assert(max_depth*max_depth*2 == nb_mat_elements)
        
        comps_repr[ogc_start[0]][ogc_start[1] : ogc_end[1] + 1 ] = get_padded_initial_constrain_matrix(program_json, schedule_json, comp_name, max_depth).flatten().tolist()
                              
        ogv_start = comps_placeholders_indices_dict[c_code+'-OgConstraintVectorStart']
        
        ogv_end = comps_placeholders_indices_dict[c_code+'-OgConstraintVectorEnd']
        
        nb_mat_elements = ogv_end[1] - ogv_start[1] + 1
        
        comps_repr[ogv_start[0]][ogv_start[1] : ogv_end[1] + 1 ] = get_padded_second_side_of_the_constraint_equations_original(program_json, schedule_json, comp_name, max_depth)
        
        c_start = comps_placeholders_indices_dict[c_code+'-ConstraintMatrixStart']
        
        c_end = comps_placeholders_indices_dict[c_code+'-ConstraintMatrixEnd']
        
        nb_mat_elements = c_end[1] - c_start[1] + 1

        assert(max_depth*max_depth*2 == nb_mat_elements)
        
        comps_repr[c_start[0]][ c_start[1] : c_end[1] + 1 ] = get_padded_transformed_constrain_matrix(program_json, schedule_json, comp_name, max_depth).flatten().tolist()
        

    # Fill the loop representation
    # Initialization
    loop_schedules_dict = dict()
    for loop_name in program_json["iterators"]:
        loop_schedules_dict[loop_name] = dict()
        loop_schedules_dict[loop_name]["tiled"] = 0
        loop_schedules_dict[loop_name]["tile_factor"] = 0
        loop_schedules_dict[loop_name]["unrolled"] = 0
        loop_schedules_dict[loop_name]["unroll_factor"] = 0
        loop_schedules_dict[loop_name]["shifted"] = 0
        loop_schedules_dict[loop_name]["shift_factor"] = 0
        loop_schedules_dict[loop_name]["parallelized"] = 0
        loop_schedules_dict[loop_name]["fused"] = 0

    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_schedule_dict = schedule_json[comp_name]
        
        # Check whether tiling was applied 
        if comp_schedule_dict["tiling"]:
            for tiled_loop_index, tiled_loop in enumerate(
                comp_schedule_dict["tiling"]["tiling_dims"]
            ):
                loop_schedules_dict[tiled_loop]["tiled"] = 1
                assert (loop_schedules_dict[tiled_loop]["tile_factor"] == 0 or loop_schedules_dict[tiled_loop]["tile_factor"] == int(
                    comp_schedule_dict["tiling"]["tiling_factors"][tiled_loop_index]
                ))
                loop_schedules_dict[tiled_loop]["tile_factor"] = int(
                    comp_schedule_dict["tiling"]["tiling_factors"][tiled_loop_index]
                )
                
        # Check whether unrolling was applied 
        if comp_schedule_dict["unrolling_factor"]:
            comp_innermost_loop = get_comp_iterators_from_tree_struct(schedule_json, comp_name)[-1]
#             comp_innermost_loop = computations_dict[comp_name]["iterators"][-1]
            loop_schedules_dict[comp_innermost_loop]["unrolled"] = 1
                
            assert (loop_schedules_dict[comp_innermost_loop]["unroll_factor"] == 0 or                                                                                           loop_schedules_dict[comp_innermost_loop]["unroll_factor"] == int(comp_schedule_dict["unrolling_factor"]))
            
            loop_schedules_dict[comp_innermost_loop]["unroll_factor"] = int(comp_schedule_dict["unrolling_factor"])
            
        # Check whether parallelization was applied 
        if comp_schedule_dict["parallelized_dim"]:
            loop_schedules_dict[comp_schedule_dict["parallelized_dim"]]["parallelized"] = 1
        
        
        if comp_schedule_dict['shiftings']:
            for shifting in comp_schedule_dict['shiftings']: 
                loop_schedules_dict[shifting[0]]["shifted"] = 1
                loop_schedules_dict[shifting[0]]["shift_factor"] = shifting[1]
        
    # Check whether fusion was applied 
    if "fusions" in schedule_json and schedule_json["fusions"]:
        for fusion in schedule_json["fusions"]:
            fused_loop1 = computations_dict[fusion[0]]["iterators"][fusion[2]]
            fused_loop2 = computations_dict[fusion[1]]["iterators"][fusion[2]]
            loop_schedules_dict[fused_loop1]["fused"] = 1
            loop_schedules_dict[fused_loop2]["fused"] = 1
            
    program_iterators = get_comp_iterators_from_tree_struct(schedule_json, comp_name)
    # Get the index of each feature in the loop representation and replace it with the the information obtained from the schedule
    for loop_name in program_json["iterators"]:
        l_code = "L" + loop_name

        p_index = loops_placeholders_indices_dict[l_code + "Parallelized"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
            "parallelized"
        ]

        p_index = loops_placeholders_indices_dict[l_code + "Tiled"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["tiled"]
        p_index = loops_placeholders_indices_dict[l_code + "TileFactor"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
            "tile_factor"
        ]

        p_index = loops_placeholders_indices_dict[l_code + "Unrolled"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["unrolled"]
        p_index = loops_placeholders_indices_dict[l_code + "UnrollFactor"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
            "unroll_factor"
        ]
        
        p_index = loops_placeholders_indices_dict[l_code + "Shifted"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["shifted"]
        p_index = loops_placeholders_indices_dict[l_code + "ShiftFactor"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name][
            "shift_factor"
        ]
        
        p_index = loops_placeholders_indices_dict[l_code + "Fused"]
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]["fused"]
    # Check if any iterators were removed because of fusion
    if (len(program_json["iterators"])>len(program_iterators)):
        removed_iterators = list(set(program_json["iterators"]) - set(program_iterators))
        for loop_name in removed_iterators:
            l_code = "L" + loop_name

            p_index = loops_placeholders_indices_dict[l_code + "Parallelized"]
            loops_repr[p_index[0]][p_index[1]] = 0

            p_index = loops_placeholders_indices_dict[l_code + "Tiled"]
            loops_repr[p_index[0]][p_index[1]] = 0
            p_index = loops_placeholders_indices_dict[l_code + "TileFactor"]
            loops_repr[p_index[0]][p_index[1]] = 0

            p_index = loops_placeholders_indices_dict[l_code + "Unrolled"]
            loops_repr[p_index[0]][p_index[1]] = 0
            p_index = loops_placeholders_indices_dict[l_code + "UnrollFactor"]
            loops_repr[p_index[0]][p_index[1]] = 0
            
            p_index = loops_placeholders_indices_dict[l_code + "Shifted"]
            loops_repr[p_index[0]][p_index[1]] = 0
            p_index = loops_placeholders_indices_dict[l_code + "ShiftFactor"]
            loops_repr[p_index[0]][p_index[1]] = 0
            
            p_index = loops_placeholders_indices_dict[l_code + "Fused"]
            loops_repr[p_index[0]][p_index[1]] = 0

    
    loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr), 0)
    computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr), 0)

    return computations_tensor, loops_tensor

# A class to contain the training and validation datasets
# Parameters:
#     dataset_filename: path to training/val dataset
#     max_batch_size: maximum batch size
#     drop_sched_func: function specifying which schedules in the dataset we want to be dropped if any
#     drop_prog_func: function specifying which programs in the dataset we want to be dropped if any
#     can_set_default_eval: function spesifying which points in the dataset can be set to default evaluation TODO add section where this is justified
#     speedups_clip_func: function spesifying which speedups to clip and to what value
#     store_device: device where to store the dataset
#     train_device: device where to train the model
class Dataset_parallel:
    def __init__(
        self,
        dataset_filename,
        max_batch_size,
        drop_sched_func=None,
        drop_prog_func=None,
        can_set_default_eval=None,
        speedups_clip_func=None,
        store_device="cpu",
        train_device="cpu",
        repr_pkl_output_folder="none",
        just_load_pickled_repr=False,
        nb_processes=15
    ):

        # Structure to contain the batched inputs
        self.batched_X = []
        # Structure to contain the batched labels
        self.batched_Y = []
        # Structure to contain TODO
        self.batches_dict = dict()
        # Maximum number of nested loops in a program
        self.max_depth = 5
        # Number of all the dropped schedules
        self.nb_dropped = 0
        self.nb_dropped_random_matrix = 0
        # Number of dropped schedules due to the drop schedule function only
        self.nb_pruned = 0
        # List of dropped functions 
        self.dropped_funcs = []
#         # Saved data attributes for analysis and expirements TODO maybe remove this
#         self.batched_datapoint_attributes = []
        # number of loaded datapoints
        self.nb_datapoints = 0
        self.gpu_fitted_batches_index = -1
        
        processes_output_list = []
        if just_load_pickled_repr: #just load the existing repr
            
            for pkl_part_filename in tqdm(list(Path(repr_pkl_output_folder).iterdir())):
                pkl_part_filename = str(pkl_part_filename)
                with open(pkl_part_filename, 'rb') as f:
                    lst = pickle.load(f)
                processes_output_list.extend(lst)
        else:
            manager = multiprocessing.Manager()

            processs = []
            queues = []
            input_queue = manager.Queue()
            output_queue = manager.Queue()

            for i in range(nb_processes):
                processs.append(multiprocessing.Process(
                target=get_func_repr_task, args=[input_queue, output_queue]))
            for process in processs:
                process.start()
                
            if dataset_filename.endswith("json"):
                with open(dataset_filename, "r") as f:
                    dataset_str = f.read()
                    
                self.programs_dict = json.loads(dataset_str)
            elif dataset_filename.endswith("pkl"):
                with open(dataset_filename, "rb") as f:
                    self.programs_dict = pickle.load(f)
            functions_list = list(self.programs_dict.keys())
            random.Random(42).shuffle(functions_list)

            nb_funcs_per_process = (len(functions_list)//nb_processes)+1
            print("number of functions per process: ",nb_funcs_per_process)

            for i in range(nb_processes):
                process_programs_dict=dict(list(self.programs_dict.items())[i*nb_funcs_per_process:(i+1)*nb_funcs_per_process])
                input_queue.put((i, process_programs_dict, repr_pkl_output_folder, store_device))

            for i in range(nb_processes):
                process_id, pkl_part_filename = output_queue.get()
                with open(pkl_part_filename, 'rb') as f:
                    lst = pickle.load(f)
                processes_output_list.extend(lst)

        
        # If no function to drop schedules was specified, define a function that doesn't drop any schedule
        if drop_sched_func == None:

            def drop_sched_func(x, y):
                return False
            
        # If no function to drop programs was specified, define a function that doesn't drop any program
        if drop_prog_func == None:

            def drop_prog_func(x, y):
                return False
            
        # If no function to clip the speedup was specified, define a function that doesn't clip the speedup
        if speedups_clip_func == None:

            def speedups_clip_func(x):
                return x
        # If no function to set the default evaluation was specified, define a function that doesn't set the evaluation to a default value for any point
        if can_set_default_eval == None:

            def can_set_default_eval(x, y):
                return 0
            
        nb_all = 0
        for function_name, nb_dropped, nb_pruned, nb_datapoints, tree_footprint, local_function_dict, nb_all_local in processes_output_list:
            for node in local_function_dict['tree']["roots"]:
                tree_indices_to_device(node, train_device=store_device)
            self.batches_dict[tree_footprint] = self.batches_dict.get(tree_footprint, {'tree': local_function_dict['tree'], 'comps_tensor_list': [], 'loops_tensor_list': [ ], 'datapoint_attributes_list': [], 'comps_expr_tree_list': [], 'speedups_list': [], 'exec_time_list': [], "func_id": []})
            
            self.batches_dict[tree_footprint]['comps_tensor_list'].extend(local_function_dict['comps_tensor_list'])
            self.batches_dict[tree_footprint]['loops_tensor_list'].extend(local_function_dict['loops_tensor_list'])
            self.batches_dict[tree_footprint]['datapoint_attributes_list'].extend(local_function_dict['datapoint_attributes_list'])
            self.batches_dict[tree_footprint]["comps_expr_tree_list"].extend(local_function_dict["comps_expr_tree_list"])
            self.batches_dict[tree_footprint]['speedups_list'].extend(local_function_dict['speedups_list'])

            self.nb_dropped += nb_dropped
            self.nb_pruned += nb_pruned
            self.nb_datapoints += len(local_function_dict['speedups_list'])
            nb_all += nb_all_local
            
        max_exprs = 51

        for tree_footprint in tqdm(self.batches_dict):
            for i in range(len(self.batches_dict[tree_footprint]["comps_expr_tree_list"])):
                for j in range(len(self.batches_dict[tree_footprint]["comps_expr_tree_list"][i])):
                    self.batches_dict[tree_footprint]["comps_expr_tree_list"][i][j].extend(
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * (max_exprs - len(self.batches_dict[tree_footprint]["comps_expr_tree_list"][i][j])))
                self.batches_dict[tree_footprint]["comps_expr_tree_list"][i] = torch.tensor(
                    [self.batches_dict[tree_footprint]["comps_expr_tree_list"][i]]).float()
                
        storing_device = torch.device(store_device)

        # For each tree footprint in the dataset TODO explain what a tree footprint is
        for tree_footprint in tqdm(self.batches_dict):

            # shuffling the lists inside each footprint to avoid having batches with very low program diversity
            zipped = list(
                zip(
                    self.batches_dict[tree_footprint]["datapoint_attributes_list"],
                    self.batches_dict[tree_footprint]["comps_tensor_list"],
                    self.batches_dict[tree_footprint]["comps_expr_tree_list"],
                    self.batches_dict[tree_footprint]["loops_tensor_list"],
                    self.batches_dict[tree_footprint]["speedups_list"],
                )
            )

            random.shuffle(zipped)
            (
                self.batches_dict[tree_footprint]["datapoint_attributes_list"],
                self.batches_dict[tree_footprint]["comps_tensor_list"],
                self.batches_dict[tree_footprint]["comps_expr_tree_list"],
                self.batches_dict[tree_footprint]["loops_tensor_list"],
                self.batches_dict[tree_footprint]["speedups_list"],
            ) = zip(*zipped)
            
            # Split the data into batches of size max_batch_size
            for chunk in range( 0, len(self.batches_dict[tree_footprint]["speedups_list"]), max_batch_size, ):
                # Check GPU memory in order to avoid Out of memory error
                if ( storing_device.type == "cuda" and ( torch.cuda.memory_allocated(storing_device.index) / torch.cuda.get_device_properties(storing_device.index).total_memory )> 0.80):
                    
                    print( "GPU memory on " + str(storing_device) + " nearly full, switching to CPU memory" )
                    self.gpu_fitted_batches_index = len(self.batched_X)
                    storing_device = torch.device("cpu")
                
#                 self.batched_datapoint_attributes.append(
#                     self.batches_dict[tree_footprint]["datapoint_attributes_list"][
#                         chunk: chunk + max_batch_size
#                     ]
#                 )
                # Here we separate the comps tensor to get the transformation vectors
                x = torch.cat( self.batches_dict[tree_footprint]["comps_tensor_list"][ chunk : chunk + max_batch_size ], 0).to(storing_device)
                batch_size, num_comps, __dict__ = x.shape
                x = x.view(batch_size * num_comps, -1)
                (first_part, vectors, third_part) = seperate_vector(
                    x, num_transformations=4, pad=False
                )
                self.batched_X.append(
                    (
                        self.batches_dict[tree_footprint]["tree"],
                        first_part.to(storing_device).view(batch_size, num_comps, -1),
                        vectors.to(storing_device), # we send it with the shape (batch_size * num_comps, num vectors) to use it directly.
                        third_part.to(storing_device).view(batch_size, num_comps, -1),
                        torch.cat( self.batches_dict[tree_footprint]["loops_tensor_list"][ chunk : chunk + max_batch_size ], 0).to(storing_device),
                        torch.cat(self.batches_dict[tree_footprint]["comps_expr_tree_list"][chunk : chunk + max_batch_size],0).to(storing_device),
                    )
                )
                self.batched_Y.append(
                    torch.FloatTensor(
                        self.batches_dict[tree_footprint]["speedups_list"][
                            chunk: chunk + max_batch_size
                        ]
                    ).to(storing_device)
                )
        if self.gpu_fitted_batches_index == -1:
            self.gpu_fitted_batches_index = len(self.batched_X)
        # shuffling batches to avoid having the same footprint in consecutive batches
        zipped = list(
            zip(
                self.batched_X,
                self.batched_Y,
#                 self.batched_datapoint_attributes,
            )
        )
        random.shuffle(zipped)
        (
            self.batched_X,
            self.batched_Y,
#             self.batched_datapoint_attributes,
        ) = zip(*zipped)

        print( f"Number of datapoints {self.nb_datapoints} Number of batches {len(self.batched_Y)}" )
        print( f"Number of dropped {self.nb_dropped}, all {nb_all}" )
        del self.batches_dict
        print("memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        gc.collect()
        print("memory usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            return (self.batched_X[index], self.batched_Y[index])

    def __len__(self):
        return len(self.batched_Y)    

    
    
    
    
# Returns a representation of the tree structure of the program
def get_tree_footprint(tree):
    if "roots" in tree :
        footprint = "<R>"
        for root in tree["roots"]:
            footprint += get_tree_footprint(root)
        footprint += "</R>"
        return footprint
    footprint = "<L" + str(int(tree["loop_index"])) + ">"
    if tree["has_comps"]:
        footprint += "["
        for idx in tree["computations_indices"]:
            footprint += "C" + str(int(idx))
        footprint += "]"
    for child in tree["child_list"]:
        footprint += get_tree_footprint(child)
    footprint += "</L" + str(int(tree["loop_index"])) + ">"
    return footprint


# load the data from the .pkl or .json dataset file and splits it according to the split_ratio parameter
# Uses the Dataset class
def load_data(
    train_val_dataset_file,
    split_ratio=None,
    max_batch_size=2048,
    drop_sched_func=None,
    drop_prog_func=None,
    default_eval=None,
    speedups_clip_func=None,
    store_device="cpu",
    train_device="cpu",
):
    print("loading batches from: " + train_val_dataset_file)
    
    # create an instance of the Dataset class
    dataset = Dataset(
        train_val_dataset_file,
        max_batch_size,
        drop_sched_func,
        drop_prog_func,
        default_eval,
        speedups_clip_func,
        store_device=store_device,
        train_device=train_device,
    )
    # If no split ratio is specified use the common 80 20 split
    if split_ratio == None:
        split_ratio = 0.2
    if split_ratio > 1:
        validation_size = split_ratio
    else:
        validation_size = int(split_ratio * len(dataset))
    indices = list(range(len(dataset)))

    # Divide the dataset indecies into validation and training 
    val_batches_indices, train_batches_indices = (
        indices[:validation_size],
        indices[validation_size:],
    )
    val_batches_list = []
    train_batches_list = []
    for i in val_batches_indices:
        val_batches_list.append(dataset[i])
    for i in train_batches_indices:
        train_batches_list.append(dataset[i])
    print("Data loaded")
    print(
        "Sizes: " + str((len(val_batches_list), len(train_batches_list))) + " batches"
    )
    return (
        dataset,
        val_batches_list,
        val_batches_indices,
        train_batches_list,
        train_batches_indices,
    )

# A function to extract the transformations applied on a spesific computation in the form of a vector of tags
# Padding is added if the number of transformations is less than the maximum value of MAX_NUM_TRANSFORMATIONS
# Currently our dataset represents transformations in two different formats.
#         1- in the form of matrices from the polyhedral representation
#         2- in the form of tags for each transformation
# We generated a variaty of representations to test which one is more useful for our spesfici usage
# In this function we will be unifying all of the dataset into the tags representation 
# The tag representation is as follows:
#         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'first_skew_factor', 'second_skew_factor']
#     Where the type_of_transformation tag is:
#         - 0 for no transformation being applied
#         - 1 for loop interchange
#         - 2 for loop reversal
#         - 3 for loop skewing
        
def get_padded_transformation_tags(
    program_json, schedule_json, comp_name, max_depth=None
):
    # Extract information about the computation and the transformations that were applied from the json input
    comp_dict = program_json["computations"][comp_name]
    comp_schedule_dict = schedule_json[comp_name]
    nb_iterators = len(comp_dict["iterators"])
    loop_nest = comp_dict["iterators"][:]
    
    # Create an identity vector that represents that no transformation was applied
    identity = np.zeros((nb_iterators, nb_iterators), int)
    np.fill_diagonal(identity, 1)
    identity_tags = np.zeros((1,MAX_TAGS), dtype=np.int32)
    
    tag_factors = []
    for transformation in comp_schedule_dict['transformations_list']:
        tag_factors.append(transformation)
    
    
    # Add padding to the sequence of vectors in case the number of transformations is less than MAX_NUM_TRANSFORMATIONS+1
    tags_list = [item for sublist in tag_factors for item in sublist]
    tags_list += [0]*(MAX_NUM_TRANSFORMATIONS*MAX_TAGS - len(tags_list)) 
    
    return tags_list

# A function to retrieve information about each datapoint
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

def pad_access_matrix(access_matrix, max_depth):
    access_matrix = np.array(access_matrix)
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix]
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix]
    padded_access_matrix = np.zeros((max_depth + 1, max_depth + 2))
    padded_access_matrix[
        : access_matrix.shape[0], : access_matrix.shape[1] - 1
    ] = access_matrix[:, :-1]
    padded_access_matrix[: access_matrix.shape[0], -1] = access_matrix[:, -1]

    return padded_access_matrix

# TODO what does this function do
def isl_to_write_matrix(isl_map):
    comp_iterators_str = re.findall(r"\[(.*)\]\s*->", isl_map)[0]
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    comp_iter_names = re.findall(r"(?:\s*(\w+))+", comp_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    matrix = np.zeros([len(buf_iter_names), len(comp_iter_names) + 1])
    for i, buf_iter in enumerate(buf_iter_names):
        for j, comp_iter in enumerate(comp_iter_names):
            if buf_iter == comp_iter:
                matrix[i, j] = 1
                break
    return matrix

# TODO what does this function do
def isl_to_write_dims(
    isl_map,
):
    buffer_iterators_str = re.findall(r"->\s*\w*\[(.*)\]", isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=", "", buffer_iterators_str)
    buf_iter_names = re.findall(r"(?:\s*(\w+))+", buffer_iterators_str)
    return buf_iter_names


# returns a pandas dataframe representing the dataset
def get_results_df(
    dataset, batches_list, indices, model, log=False, train_device="cpu"
):
    df = pd.DataFrame()
    model.eval()
    torch.set_grad_enabled(False)
    all_outputs = []
    all_labels = []
    prog_names = []
    sched_names = []
    exec_times = []
    sched_strs = []
    memory_uses = []
    node_names = []
    tree_footprints = []

    for k, (inputs, labels) in tqdm(list(enumerate(batches_list))):
        original_device = labels.device
        inputs = (inputs[0], inputs[1].to(train_device), inputs[2].to(train_device))
        labels = labels.to(train_device)
        outputs = model(inputs)
        assert outputs.shape == labels.shape
        all_outputs.append(outputs)
        all_labels.append(labels)

        assert len(outputs) == len(dataset.batched_datapoint_attributes[indices[k]])
        zipped_attributes = list(zip(*dataset.batched_datapoint_attributes[indices[k]]))
        prog_names.extend(zipped_attributes[0])
        sched_names.extend(zipped_attributes[1])
        sched_strs.extend(zipped_attributes[2])
        exec_times.extend(zipped_attributes[3])
        memory_uses.extend(zipped_attributes[4])
        node_names.extend(zipped_attributes[5])
        tree_footprints.extend(zipped_attributes[6])
        inputs = (
            inputs[0],
            inputs[1].to(original_device),
            inputs[2].to(original_device),
        )
        labels = labels.to(original_device)
    preds = torch.cat(all_outputs)
    targets = torch.cat(all_labels)
    preds = preds.cpu().detach().numpy().reshape((-1,))
    preds = np.around(preds, decimals=6)
    targets = np.around(targets.cpu().detach().numpy().reshape((-1,)), decimals=6)

    assert preds.shape == targets.shape
    df["name"] = prog_names
    df["tree_struct"] = tree_footprints
    df["sched_name"] = sched_names
    df["sched_str"] = sched_strs
    df["exec_time"] = exec_times
    df["memory_use"] = list(map(float, memory_uses))
    df["node_name"] = node_names
    df["prediction"] = np.array(preds)
    df["target"] = np.array(targets)

    df["APE"] = np.abs(df.target - df.prediction) / df.target * 100
    df["sched_str"] = df["sched_str"].apply(lambda x: simplify_sched_str(x))

    return df

# Calculate the Normalized Discounted Cumulative Gain while only considiring the top rated schedule (k=1)
def function_wise_ndcg_1(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG_1=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=1)
    return pd.Series(dict(nDCG_1=score))

# Calculate the Normalized Discounted Cumulative Gain while only considiring the top 5 rated schedules (k=5)
def function_wise_ndcg_5(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG_5=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=5)
    return pd.Series(dict(nDCG_5=score))

# Calculate the Normalized Discounted Cumulative Gain while considiring all the schedules
def function_wise_ndcg_full(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(nDCG=np.nan))
    score = ndcg_score([g["target"].tolist()], [g["prediction"].tolist()], k=None)
    return pd.Series(dict(nDCG=score))

# Calculate the Spearman correlation coefficient
def function_wise_spearman(g):
    if len(g["target"]) < 2:
        return pd.Series(dict(Spearman_r=np.nan))
    score = spearmanr(g["target"], g["prediction"])[0]
    return pd.Series(dict(Spearman_r=score))

# Calculate the absolute percentage error
def function_wise_ape(g):
    score = np.mean(g["APE"])
    return pd.Series(dict(MAPE=score))


def get_scores(df):
    with tqdm(total=6) as pbar:
        df_spearman = df.groupby("name").apply(function_wise_spearman).reset_index()
        pbar.update(1)
        df_mape = df.groupby("name").apply(function_wise_ape).reset_index()
        pbar.update(1)
        df_ndcg = df.groupby("name").apply(function_wise_ndcg_full).reset_index()
        pbar.update(1)
        df_ndcg1 = df.groupby("name").apply(function_wise_ndcg_1).reset_index()
        pbar.update(1)
        df_ndcg5 = df.groupby("name").apply(function_wise_ndcg_5).reset_index()
        pbar.update(1)
        df_count = df.groupby("name").agg("count").reset_index()[["name", "sched_name"]]
        df_count.columns = ["name", "count"]
        pbar.update(1)

    scores_df = (
        df_count.merge(df_ndcg, on="name")
        .merge(df_ndcg5, on="name")
        .merge(df_ndcg1, on="name")
        .merge(df_spearman, on="name")
        .merge(df_mape, on="name")
    )
    return scores_df


def simplify_sched_str(
    sched_str,
):

    if sched_str.count("M") == 1:
        return sched_str
    comps = re.findall("C\d+", sched_str)
    comps = set(comps)

    mats = set(re.findall(r"M\({[\dC\,]+},([\d\,\-]+)", sched_str))
    comps_per_mat = {mat: [] for mat in mats}
    new_mats_str = ""
    for mat in comps_per_mat:
        for mat_part in re.findall("M\({[C\d\,]+}," + mat, sched_str):
            comps_per_mat[mat].extend(re.findall("C\d+", mat_part))
        new_mats_str += "M({" + ",".join(sorted(comps_per_mat[mat])) + "}," + mat + ")"
    return re.sub("(M\({[\dC\,]+},[\d\,\-]+\))+", new_mats_str, sched_str)



# TODO
def has_skippable_loop_1comp(
    prog_dict,
):

    program_json = prog_dict["program_annotation"]
    if not len(program_json["computations"]) == 1:
        return False
    comp_name = list(program_json["computations"].keys())[0]
    comp_dict = program_json["computations"][comp_name]
    write_buffer_id = comp_dict["write_buffer_id"]
    iterators = comp_dict["iterators"]
    write_dims = isl_to_write_dims(comp_dict["write_access_relation"])
    read_buffer_ids = [e["buffer_id"] for e in comp_dict["accesses"]]

    if len(write_dims) == len(iterators):

        if (
            len(read_buffer_ids) == 1
            and read_buffer_ids[0] == write_buffer_id
            and comp_dict["number_of_additions"] == 0
            and comp_dict["number_of_subtraction"] == 0
            and comp_dict["number_of_multiplication"] == 0
            and comp_dict["number_of_division"] == 0
        ):
            return True
        return False

    if not write_buffer_id in read_buffer_ids:
        return True

    found = False
    for access in comp_dict["accesses"]:
        if access["buffer_id"] == write_buffer_id and not access_is_stencil(access):
            found = True
            break
    if not found:
        if write_dims[-1] != iterators[-1]:
            return True

    for access in comp_dict["accesses"]:
        if access["buffer_id"] == write_buffer_id and access_is_stencil(access):
            return False

    read_dims_bools = []
    for access in comp_dict["accesses"]:
        read_dims_bools.append(np.any(access["access_matrix"], axis=0))
    read_dims_bools = np.any(read_dims_bools, axis=0)
    read_iterators = [
        iterators[i]
        for i, is_used in enumerate(read_dims_bools[:-1])
        if is_used == True
    ]
    used_iterators = set(write_dims + read_iterators)
    if len(used_iterators) == len(iterators):
        return False

    if iterators[-1] in used_iterators:
        if len(comp_dict["accesses"]) > 2:
            return False

    return True


def sched_is_prunable(program_json, schedule_json):
    reg = ""
    comp_names = [
        n
        for n in schedule_json.keys()
        if not n in ["unfuse_iterators", "tree_structure", "execution_times", "fusions", "sched_str", "legality_check", "exploration_method"]
    ]
    for name in comp_names:
        innermost_loop = len(program_json["computations"][name]["iterators"])-1
        reg += f"{{{name}}}:P\(L{innermost_loop}\)U.*|"
    if(len(comp_names)>0):
        reg = reg[:-1]
    
    schedule_str = get_schedule_str_for_pruning(program_json, schedule_json)
    if re.search(reg, schedule_str):
        return True
    reg = ""
    
    for name in comp_names:
        innermost_loop = len(program_json["computations"][name]["iterators"])-1
        reg += f"{{{name}}}:P\(L{innermost_loop}\)T2\(L{innermost_loop-2},L{innermost_loop-1}.*|"
    
    if(len(comp_names)>0):
        reg = reg[:-1]
    if re.search(reg, schedule_str):
        return True                                                                                                                               
    return False

def can_set_default_eval(program_json, schedule_json):
    reg = ""
    comp_names = [
        n
        for n in schedule_json.keys()
        if not n in ["unfuse_iterators", "tree_structure", "execution_times", "fusions", "sched_str", "legality_check", "exploration_method"]
    ]
    for name in comp_names:
        innermost_loop = len(program_json["computations"][name]["iterators"])-1
        reg += f"{{{name}}}:P\(L{innermost_loop}\)({{|$)|"
    if(len(comp_names)>0):
        reg = reg[:-1]
    
    schedule_str = get_schedule_str_for_pruning(program_json, schedule_json)
    if re.search(reg, schedule_str):    
        return 0.01
    
    return 0

def access_is_stencil(access):
    return np.any(access["access_matrix"], axis=0)[-1]

# Solving the Linear Diophantine equation & finding basic solution (sigma & gamma) for : f_i* sigma - f_j*gamma = 1
# Used to get skewing parameters
def linear_diophantine_default(f_i, f_j):
    n1 = abs(f_i)
    n2 = abs(f_j)
    
    while(n1 != n2):
        if(n1 > n2):
            n1 -=  n2
        else:
            n2 -=  n1
            
    # Update f_i and f_j to equivalent but prime between themselfs value
    f_i = f_i / n1
    f_j = f_j / n1
    
    found = False
    gamma = 0
    sigma = 1
    
    if (f_j == 1) or (f_i == 1):
        gamma = f_i - 1
        sigma = 1
        # Since sigma = 1  then
        # f_i - gamma * f_j = 1 & using the previous condition :
        #  - f_i = 1 : then gamma = 0 (f_i-1) is enough
        #  - f_j = 1 : then gamma = f_i -1  
    else:
        if (f_j == -1) and (f_i > 1):
            gamma = 1
            sigma = 0
        else:
            # General case : solving the Linear Diophantine equation & finding basic solution (sigma & gamma) for : f_i* sigma - f_j*gamma = 1
            i = 0
            while (i < 100) and (not found):
                if ((sigma * f_i) % abs(f_j)) == 1:
                    found = True
                else:
                    sigma += 1
                    i += 1
            if not found:
                # Detect infinite loop and prevent it in case where f_i and f_j are not prime between themselfs
                print("Error cannof find solution to diophantine equation")
                return
            gamma = ((sigma * f_i) - 1) / f_j
    return gamma, sigma

def wrongly_set_to_default_schedule(program_dict, schedule_index):
    
    schedule_dict = program_dict["schedules_list"][schedule_index]
    if len(schedule_dict["execution_times"]) == 1:
        speed_up = program_dict["initial_execution_time"] / schedule_dict["execution_times"][0]
        
        if (speed_up > 0.00099 and speed_up < 0.00101) and (can_set_default_eval(program_dict["program_annotation"], schedule_dict) == 0):
                return True
    return False

#TODO
def speedup_clip(speedup):
    if speedup < 0.01:
        speedup = 0.01
    return speedup

#TODO
def drop_program(prog_dict, prog_name):
#     print(prog_name[8:])
    if len(prog_dict["schedules_list"]) < 2:
        return True
    if ( 750000 <= int(prog_name[8:]) and int(prog_name[8:])<=752600 ):
    
#         print("Found an random matrix program", prog_name)
        return True
    if has_skippable_loop_1comp(prog_dict):
        return True
    if (
        "node_name" in prog_dict and prog_dict["node_name"] == "lanka24"
    ):  # drop if we the program is run by lanka24 (because its measurements are inacurate)
        return True
    return False

def drop_schedule(prog_dict, schedule_index):
    schedule_json = prog_dict["schedules_list"][schedule_index]
    if (not schedule_json["execution_times"]) or min(
        schedule_json["execution_times"]
    ) < 0:  # exec time is set to -1 on datapoints that are deemed noisy, or if list empty
        return True
    if sched_is_prunable(prog_dict["program_annotation"], schedule_json):
            return True
    if wrongly_set_to_default_schedule(prog_dict, schedule_index):
        return True

    return False

def get_involved_comps(node):
        result = []
        if(len(node)==0): 
            return result
        for comp in node["computations_list"]:
            result.append(comp)
        for child in node["child_list"]:
            for comp in get_involved_comps(child):
                result.append(comp)
        return result
def get_comp_iterators_from_tree_struct(schedule_json, comp_name):
    tree = schedule_json["tree_structure"]
    level = tree
    iterators = []
    to_explore = []
    # only add the root that contains the computation we are looking for
    for root in tree["roots"]:
        if (comp_name in get_involved_comps(root)):
            to_explore.append(root)
    
    while(to_explore):
        level = to_explore.pop(0)
        if(comp_name in get_involved_comps(level)):
            iterators.append(level['loop_name'])
            
        for element in level["child_list"]:
            to_explore.append(element)
    
    return iterators
def get_expr_repr(expr, comp_type):
        expr_vector = []
        if(expr == "add"):
            expr_vector = [1, 0, 0, 0, 0, 0, 0, 0]
        elif(expr == "sub"):
            expr_vector = [0, 1, 0, 0, 0, 0, 0, 0]
        elif(expr == "mul"):
            expr_vector = [0, 0, 1, 0, 0, 0, 0, 0]
        elif(expr == "div"):
            expr_vector = [0, 0, 0, 1, 0, 0, 0, 0]
        elif(expr == "sqrt"):
            expr_vector = [0, 0, 0, 0, 1, 0, 0, 0]
        elif(expr == "min"):
            expr_vector = [0, 0, 0, 0, 0, 1, 0, 0]
        elif(expr == "max"):
            expr_vector = [0, 0, 0, 0, 0, 0, 1, 0]
        else:
            expr_vector = [0, 0, 0, 0, 0, 0, 0, 1]
        
        comp_type_vector = []
        if(comp_type == "int32"):
            comp_type_vector = [1, 0, 0]
        elif(comp_type == "float32"):
            comp_type_vector = [0, 1, 0]
        elif(comp_type == "float64"):
            comp_type_vector = [0, 0, 1]
            
        return expr_vector + comp_type_vector

def get_tree_expr_repr(node, comp_type):
        expr_tensor = []
        if node["children"] != []:
            for child_node in node["children"]:
                expr_tensor.extend(get_tree_expr_repr(child_node, comp_type))
        expr_tensor.append(get_expr_repr(node["expr_type"], comp_type))

        return expr_tensor
    
def get_padded_initial_constrain_matrix(program_json, schedule_json, comp_name, max_depth):
    
    iterators_list = program_json["computations"][comp_name]["iterators"]
    result = []
    for i in iterators_list:
        for j in range(2):
            if j == 0 :
                result.append(format_bound(i, program_json["iterators"][i]["lower_bound"], iterators_list, True))
            else:
                result.append(format_bound(i, program_json["iterators"][i]["upper_bound"], iterators_list, False))
    result = np.array(result)            
    result = np.pad(
        result,
        [
            (0, (max_depth)*2 - result.shape[0]),
            (0, max_depth - result.shape[1]),
        ],
        mode="constant",
        constant_values=0,
    )
    return result

# check whether the string contains an integer and return true if so
def is_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

# returns a vector that represents the right hand sise of teh constraint matrix inequalities
# returns b where: Ax <= b and A being the constarint matrix
def get_padded_second_side_of_the_constraint_equations_original(program_json, schedule_json, comp_name, max_depth):
    iterators_list = program_json["computations"][comp_name]["iterators"]
    result = []
    for it in iterators_list:
        if(is_int(program_json["iterators"][it]["lower_bound"])):
            result.append(int(program_json["iterators"][it]["lower_bound"]))
        else:
            result.append(0)
        if(is_int(program_json["iterators"][it]["upper_bound"])):
            result.append(int(program_json["iterators"][it]["upper_bound"]))
        else:
            result.append(0)
    result = result + [0]*(max_depth*2-len(result))
    return result
                              
def get_padded_transformed_constrain_matrix(program_json, schedule_json, comp_name, max_depth):
    iterators_list = program_json["computations"][comp_name]["iterators"]
    transformation_matrix = get_transformation_matrix(program_json, schedule_json, comp_name, max_depth)
    result = []
    for i in iterators_list:
        for j in range(2):
            if j == 0 :
                result.append(format_bound(i, program_json["iterators"][i]["lower_bound"], iterators_list, True))
            else:
                result.append(format_bound(i, program_json["iterators"][i]["upper_bound"], iterators_list, False))
    inverse = np.linalg.inv(transformation_matrix)
    result = np.matmul(result, inverse)
    result = np.array(result)
    result = np.pad(
        result,
        [
            (0, (max_depth)*2 - result.shape[0]),
            (0, max_depth - result.shape[1]),
        ],
        mode="constant",
        constant_values=0,
    )
    
    return result

def format_bound(iterator_name, bound, iterators_list, is_lower):
    output = []
    for i in iterators_list:
        if i == iterator_name:
            if is_lower :
                output.append(-1)
            else:
                output.append(1)
        elif (i == bound):
            if is_lower :
                output.append(1)
            else:
                output.append(-1)
        else:
            output.append(0)
    return output

def get_trasnformation_matrix_from_vector(transformation, matrix_size):
    matrix = np.identity(matrix_size)
    
    if (transformation[0] == 1):
        assert(transformation[1] < matrix_size and transformation[2] < matrix_size)
        matrix[transformation[1], transformation[2]] = 1
        matrix[transformation[1], transformation[1]] = 0
        matrix[transformation[2], transformation[1]] = 1
        matrix[transformation[2], transformation[2]] = 0
        
    elif (transformation[0] == 2):
        assert(transformation[3] < matrix_size)
        matrix[transformation[3], transformation[3]] = -1
        
    elif (transformation[0] == 3):
        assert(transformation[4] < matrix_size and transformation[5] < matrix_size)
        matrix[transformation[4], transformation[4]] = transformation[6]
        matrix[transformation[4], transformation[5]] = transformation[7]
    
    return matrix
# transform the vectors into a series of matrices
def get_transformation_matrix(
    program_json, schedule_json, comp_name, max_depth=None
):
    nb_iterators = len(program_json["computations"][comp_name]["iterators"])
    final_transformation = np.identity(nb_iterators)
    for transformation in schedule_json[comp_name]["transformations_list"]:
        matrix = get_trasnformation_matrix_from_vector(transformation, nb_iterators)
        final_transformation = np.matmul(matrix, final_transformation)
    return final_transformation

def get_schedule_str_for_pruning(program_json, sched_json):
    comp_name = [
        n
        for n in sched_json.keys()
        if not n in ["unfuse_iterators", "tree_structure", "execution_times", "fusions", "sched_str", "legality_check", "exploration_method"]
    ]
    sched_str = ""
#     print(f"starting for new schedules in program: {program_json}")
    for name in comp_name:
        # can probably use the feature in prog json
        transf_loop_nest = program_json["computations"][name]["iterators"].copy()
        schedule = sched_json[name]
        sched_str += '{' + name + '}:'
#         print(f"transf_loop_nest for computation: {name} is {transf_loop_nest}")
        if schedule["parallelized_dim"]:
            
            dim_index = transf_loop_nest.index(schedule["parallelized_dim"])
            sched_str += "P(L" + str(dim_index) + ")"

        if schedule["tiling"]:
            if schedule["tiling"]["tiling_depth"] == 2:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                second_dim = schedule["tiling"]["tiling_dims"][1]
                
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                second_factor = schedule["tiling"]["tiling_factors"][1]
                sched_str += (
                    "T2(L"
                    + str(first_dim_index)
                    + ",L"
                    + str(second_dim_index)
                    + ","
                    + str(first_factor)
                    + ","
                    + str(second_factor)
                    + ")"
                )
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[i : i + 1] = first_dim + "_outer", second_dim + "_outer"
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[i : i + 1] = first_dim + "_inner", second_dim + "_inner"
            else:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                second_dim = schedule["tiling"]["tiling_dims"][1]
                third_dim = schedule["tiling"]["tiling_dims"][2]
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                third_dim_index = transf_loop_nest.index(third_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                second_factor = schedule["tiling"]["tiling_factors"][1]
                third_factor = schedule["tiling"]["tiling_factors"][2]
                sched_str += (
                    "T3(L"
                    + str(first_dim_index)
                    + ",L"
                    + str(second_dim_index)
                    + ",L"
                    + str(third_dim_index)
                    + ","
                    + str(first_factor)
                    + ","
                    + str(second_factor)
                    + ","
                    + str(third_factor)
                    + ")"
                )
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[i : i + 1] = (
                    first_dim + "_outer",
                    second_dim + "_outer",
                    third_dim + "_outer",
                )
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[i : i + 1] = (
                    first_dim + "_inner",
                    second_dim + "_inner",
                    third_dim + "_inner",
                )
                transf_loop_nest.remove(third_dim)

        if schedule["unrolling_factor"]:
            dim_index = len(transf_loop_nest) - 1
            dim_name = transf_loop_nest[-1]
            sched_str += "U(L" + str(dim_index) + "," + schedule["unrolling_factor"] + ")"
            transf_loop_nest[dim_index : dim_index + 1] = (
                dim_name + "_Uouter",
                dim_name + "_Uinner",
            )
    return sched_str
# returns a string representation of a schedule and the transformations applied in it
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
        sched_str += '{' + name + '}:'

        for transformation in schedule["transformations_list"]:

            if (transformation[0] == 1):
                sched_str += "I(L" + str(transformation[1]) + ",L" + str(transformation[2]) + ")"
                
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
            if schedule["tiling"]["tiling_depth"] == 2:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                second_dim = schedule["tiling"]["tiling_dims"][1]
                
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                second_factor = schedule["tiling"]["tiling_factors"][1]
                sched_str += (
                    "T2(L"
                    + str(first_dim_index)
                    + ",L"
                    + str(second_dim_index)
                    + ","
                    + str(first_factor)
                    + ","
                    + str(second_factor)
                    + ")"
                )
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[i : i + 1] = first_dim + "_outer", second_dim + "_outer"
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[i : i + 1] = first_dim + "_inner", second_dim + "_inner"
            else:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                second_dim = schedule["tiling"]["tiling_dims"][1]
                third_dim = schedule["tiling"]["tiling_dims"][2]
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                third_dim_index = transf_loop_nest.index(third_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                second_factor = schedule["tiling"]["tiling_factors"][1]
                third_factor = schedule["tiling"]["tiling_factors"][2]
                sched_str += (
                    "T3(L"
                    + str(first_dim_index)
                    + ",L"
                    + str(second_dim_index)
                    + ",L"
                    + str(third_dim_index)
                    + ","
                    + str(first_factor)
                    + ","
                    + str(second_factor)
                    + ","
                    + str(third_factor)
                    + ")"
                )
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[i : i + 1] = (
                    first_dim + "_outer",
                    second_dim + "_outer",
                    third_dim + "_outer",
                )
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[i : i + 1] = (
                    first_dim + "_inner",
                    second_dim + "_inner",
                    third_dim + "_inner",
                )
                transf_loop_nest.remove(third_dim)

        if schedule["unrolling_factor"]:
            dim_index = len(transf_loop_nest) - 1
            dim_name = transf_loop_nest[-1]
            sched_str += "U(L" + str(dim_index) + "," + schedule["unrolling_factor"] + ")"
            transf_loop_nest[dim_index : dim_index + 1] = (
                dim_name + "_Uouter",
                dim_name + "_Uinner",
            )
    return sched_str

def seperate_vector(
    X: torch.Tensor, num_transformations: int = 4, pad: bool = True, pad_amount: int = 5
) -> torch.Tensor:
    batch_size, _ = X.shape
    first_part = X[:, :33]
    second_part = X[:, 33 : 33 + MAX_TAGS * num_transformations]
    third_part = X[:, 33 + MAX_TAGS * num_transformations :]
    vectors = []
    for i in range(num_transformations):
        vector = second_part[:, MAX_TAGS * i : MAX_TAGS * (i + 1)].reshape(batch_size, 1, -1)
        vectors.append(vector)

    if pad:
        for i in range(pad_amount):
            vector = torch.zeros_like(vector)
            vectors.append(vector)
    return (first_part, torch.cat(vectors[0:], dim=1), third_part)


def load_pickled_repr(repr_pkl_output_folder=None,max_batch_size = 1024, store_device="cpu", train_device="cpu"):
    
    print("loading existing batches from: " + repr_pkl_output_folder)
    print(store_device)
    print(train_device)
    dataset = Dataset_parallel(None, max_batch_size, None, repr_pkl_output_folder=repr_pkl_output_folder, just_load_pickled_repr=True, store_device=store_device, train_device=train_device)
    
    indices = list(range(len(dataset)))
 
    batches_list=[]
    for i in indices:
        batches_list.append(dataset[i])
        
    print("Data loaded")
    print("Size: "+str(len(batches_list))+" batches")
    return dataset, batches_list, indices

def tree_indices_to_device(node, train_device):
    node['loop_index'] = node['loop_index'].to(train_device, non_blocking=True)
    if 'computations_indices' in node:
        node['computations_indices'] = node['computations_indices'].to(
            train_device, non_blocking=True)
    for child in node['child_list']:
        tree_indices_to_device(child, train_device)
        

def get_func_repr_task(input_q, output_q):
    #     print('waiting for task')
    process_id, programs_dict, repr_pkl_output_folder, train_device = input_q.get()
    function_name_list = list(programs_dict.keys())
    nb_dropped = 0
    nb_pruned = 0
    nb_datapoints = 0
    nb_all=0
    cpt = 0
    dropped_funcs = []
    local_list = []
    for function_name in tqdm(function_name_list):
        nb_dropped = 0
        nb_pruned = 0
        nb_all=0
        nb_all += len(
                programs_dict[function_name]["schedules_list"]
            )
        # Check whether this function should be dropped
        if drop_program(programs_dict[function_name], function_name):
            nb_dropped += len(
                programs_dict[function_name]["schedules_list"]
            )
            dropped_funcs.append(function_name)
            continue
            
        # Get the JSON representation of the program features
        program_json = programs_dict[function_name]["program_annotation"]
        # Extract the representation template for the datapoints
        try:
            (
                prog_tree,
                comps_repr_templates_list,
                loops_repr_templates_list,
                comps_placeholders_indices_dict,
                loops_placeholders_indices_dict,
                comps_expr_repr_templates_list
            ) = get_representation_template(
                programs_dict[function_name],
                max_depth=5,
                train_device=train_device,
            )
        
        except (NbAccessException, LoopsDepthException):
            # If one of the two exceptions was raised, we drop all the schedules for that program and skip to the next program.
            nb_dropped += len(
                programs_dict[function_name]["schedules_list"]
            )
            continue
        
        # Get the initial execution time for the program to calculate the speedups (initial exec time / transformed exec time)
        program_exec_time = programs_dict[function_name][
            "initial_execution_time"
        ]
        # Get the program tree footprint
        tree_footprint = get_tree_footprint(prog_tree)
        
        local_function_dict = {
            "tree": prog_tree,
            "comps_tensor_list": [],
            "comps_expr_tree_list": [],
            "loops_tensor_list": [],
            "datapoint_attributes_list": [],
            "speedups_list": [],
            "exec_time_list": [],
            "func_id": [],
        }
        # For each schedule (sequence of transformations) collected for this function
        for schedule_index in range(len(programs_dict[function_name]['schedules_list'])):
            
            # Get the schedule JSON representation
            schedule_json = programs_dict[function_name]['schedules_list'][schedule_index]
            
            # Get the transformed execution time
            sched_exec_time = np.min(schedule_json['execution_times'])
            
            # Check if this schedule should be dropped
            if drop_schedule(programs_dict[function_name], schedule_index) or (not sched_exec_time):
                nb_dropped += 1
                nb_pruned += 1
                continue
            # Calculate the speed up obtained from applying the list of transformations spesified by the schedule
            sched_speedup = program_exec_time / sched_exec_time
            
            # Check whether we can set a default value for this speedup through the can_set_default_eval function.
            def_sp = can_set_default_eval(programs_dict[function_name]["program_annotation"], schedule_json)
            
            # If the function returns 0, this means no default value was spesified
            if def_sp > 0:
                sched_speedup = def_sp
                
            # Check whether we can clip the obtained speedup
            sched_speedup = speedup_clip(sched_speedup)
            
            # Fill the obtained template with the corresponsing schedule features
            try:
                comps_tensor, loops_tensor = get_schedule_representation(
                    program_json,
                    schedule_json,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                    max_depth=5,
                )
            except NbTranformationException:
                    # If the number of transformations exceeded the specified max, we skip this schedule
                    self.nb_dropped += 1 
                    continue
            except RandomMatrix :
                self.nb_dropped += 1
                self.nb_dropped_random_matrix += 1
                continue
            # Get information about this datapoint (memory use, execution time...)
            datapoint_attributes = get_datapoint_attributes(
                function_name, programs_dict[function_name], schedule_index, tree_footprint)
                
            # Add each part of the input to the local_function_dict to be sent to the parent process
            local_function_dict['comps_tensor_list'].append(comps_tensor)
            local_function_dict['loops_tensor_list'].append(loops_tensor)
            local_function_dict['datapoint_attributes_list'].append(
                datapoint_attributes)
            local_function_dict["comps_expr_tree_list"].append(
                comps_expr_repr_templates_list
            )
            local_function_dict['speedups_list'].append(sched_speedup)
            nb_datapoints += 1
        
        local_list.append((function_name, nb_dropped, nb_pruned,
                           nb_datapoints, tree_footprint, local_function_dict,nb_all))
    
    pkl_part_filename = repr_pkl_output_folder + '/pickled_representation_part_'+str(process_id)+'.pkl'
    with open(pkl_part_filename, 'wb') as f:
        pickle.dump(local_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    output_q.put((process_id, pkl_part_filename))
    
def load_data_parallel(train_val_dataset_file, max_batch_size=2048, nb_processes=15, repr_pkl_output_folder=None, overwrite_existing_pkl=False, store_device="none", train_device="none" ):
    torch.set_printoptions(threshold=10_000)
    if Path(repr_pkl_output_folder).is_dir() and overwrite_existing_pkl:
        shutil.rmtree(repr_pkl_output_folder)
        print('deleted existing folder ',repr_pkl_output_folder)
        
    Path(repr_pkl_output_folder).mkdir(parents=True, exist_ok=False)
    print('Created folder ',repr_pkl_output_folder)
    
    print("loading batches from: "+train_val_dataset_file)
    dataset = Dataset_parallel(train_val_dataset_file, max_batch_size, nb_processes=nb_processes, repr_pkl_output_folder=repr_pkl_output_folder, store_device=store_device, train_device=store_device)
    indices = list(range(len(dataset)))
 
    batches_list=[]
    for i in indices:
        batches_list.append(dataset[i])
        
    print("Data loaded")
    print("Size: "+str(len(batches_list))+" batches")
    return dataset, batches_list, indices, dataset.gpu_fitted_batches_index