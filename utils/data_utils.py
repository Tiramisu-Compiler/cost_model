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

# An exception to limit the maximum number of allowed transformations 
class NbTranformationException(Exception):
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
MAX_TAGS = 16

# Maximum depth of a loop nest for each computation
MAX_DEPTH = 7

# Maximum length of expressions in the dataset
MAX_EXPR_LEN = 62

# Creates a template for the input representation
def get_representation_template(program_annotation, tree_annotation, train_device="cpu"):
    # Set the max and min number of accesses allowed 
    max_accesses = 15
    min_accesses = 0

    comps_repr_templates_list = []
    comps_indices_dict = dict()
    comps_placeholders_indices_dict = dict()

    # Get the program JSON represenation
    program_json = program_annotation
    
    # Get the computations (program statements) dictionary and order them according to the absolute_order attribute
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )
    # For each computation in the program
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        # Check if the computation accesses conform to the minimum and maximum allowed
        if len(comp_dict["accesses"]) > max_accesses:
            raise NbAccessException
        
        if len(comp_dict["accesses"]) < min_accesses:
            raise NbAccessException
        
        # Check if the number of iterators for this computation doesn't surpass the maximum allowed
        if len(comp_dict["iterators"]) > MAX_DEPTH:
            raise LoopsDepthException
    
        comp_repr_template = []
        comp_repr_template.append(+comp_dict["comp_is_reduction"])

        iterators_repr = []
        
        # Add a representation of each loop of this computation
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
            iterator_dict = program_json["iterators"][iterator_name]
            # Create a unique code for each loop
            c_code = "C" + str(comp_index)
            l_code = c_code + "-L" + str(iter_i)
            
            # Add a placeholder for transformations applied to this loop
            iterators_repr.extend(
                [
                    l_code + "upper_bound",
                    l_code + "lower_bound",
                ]
            )
        
        iterator_repr_size = int(len(iterators_repr) / len(comp_dict["iterators"]))
        
        # Add padding incase the number of loops is lower than the max
        iterators_repr.extend(
            [0] * iterator_repr_size * (MAX_DEPTH - len(comp_dict["iterators"]))
        )
                              
        # Add the loop representation to the computation vector 
        comp_repr_template.extend(iterators_repr)
        
        # Pad the write access matrix and add it to the representation
        padded_write_matrix = pad_access_matrix(
            isl_to_write_matrix(comp_dict["write_access_relation"])
        )
        write_access_repr = [
            comp_dict["write_buffer_id"] + 1
        ] + padded_write_matrix.flatten().tolist()
        comp_repr_template.extend(write_access_repr)

        # Pad the read access matrix and add it to the representation 
        read_accesses_repr = []
        for read_access_dict in comp_dict["accesses"]:
            read_access_matrix = pad_access_matrix(
                read_access_dict["access_matrix"]
            )
            read_access_repr = (
                [+read_access_dict["access_is_reduction"]]
                + [read_access_dict["buffer_id"] + 1]
                + read_access_matrix.flatten().tolist()
            )
            read_accesses_repr.extend(read_access_repr)
        access_repr_len = (MAX_DEPTH + 1) * (MAX_DEPTH + 2) + 1 + 1
        read_accesses_repr.extend(
            [0] * access_repr_len * (max_accesses - len(comp_dict["accesses"]))
        )
        comp_repr_template.extend(read_accesses_repr)
        
        # Add the representation of this computation to the list of containing all computations
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
                l_code + "upper_bound",
                l_code + "lower_bound",
            ]
        )
        
        # Create a mapping between the features and their position in the representation
        loops_repr_templates_list.append(loop_repr_template)
        loops_indices_dict[loop_name] = loop_index
        for j, element in enumerate(loop_repr_template):
            if isinstance(element, str):
                loops_placeholders_indices_dict[element] = (loop_index, j)
    
    # Add necessary attributes to the tree_structure
    prog_tree = update_tree_atributes(tree_annotation, loops_indices_dict, comps_indices_dict, train_device="cpu")
    
    return (
        prog_tree,
        comps_repr_templates_list,
        loops_repr_templates_list,
        comps_placeholders_indices_dict,
        loops_placeholders_indices_dict
    )

# Change the structure of the tree annotations to contain a uinque index for each loop and a has_comps boolean
# This is used to prepare for the recusive embedding of the program during the training
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

def get_nb_iterators(tree_annotation, computation):
    """
    Given a tree structure and computation, return the number of iterators containing this computation in the tree
    TODO: Improve the efficieny of the implementation
    """
    nb_iterators = 0
    for root in tree_annotation["roots"]:
        if computation in get_node_computations(root):
            iterator =  root
            nb_iterators += 1
            while iterator:
                next_iterator = None
                for child in iterator["child_list"]:
                    if computation in get_node_computations(child):
                        next_iterator = child
                        nb_iterators += 1
                iterator = next_iterator
            break
    
    return nb_iterators
    
    
def get_node_computations(node):
    node_outputs = []
    
    if node["computations_list"] != []:
        node_outputs.extend(node["computations_list"])
        
    for child_node in node["child_list"]:
        node_outputs.extend(get_node_computations(child_node))
    
    return node_outputs
    
def get_computations_per_root(tree_annotation):
    """ 
    Given a tree structure, return a list of lists of computations where each list represents the computations in a single root.
    The roots are ordered according to their position in the tree structure
    """
    output_comps = []
    for root in tree_annotation["roots"]:
        root_comps = get_node_computations(root)
        assert root_comps
        output_comps.append(root_comps)
    assert len(tree_annotation["roots"]) == len(output_comps)
    return output_comps  
    
# Fill the representation template with the corresponsing schedule features 
def get_schedule_representation(
    program_json,
    schedule_json,
    comps_repr_templates_list,
    loops_repr_templates_list,
    comps_placeholders_indices_dict,
    loops_placeholders_indices_dict,   
):
    
    # Create a copy of the templates to avoid modifying the values for other schedules
    comps_repr = copy.deepcopy(comps_repr_templates_list)
    loops_repr = copy.deepcopy(loops_repr_templates_list)
    comps_expr_repr = []
    
    # Get an ordered list of computations from the program JSON
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )
    
    # For each computation
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = program_json["computations"][comp_name]
        
        # Get the computation expression representation
        expr_dict = comp_dict["expression_representation"]
        comp_type = comp_dict["data_type"]
        expression_representation = get_tree_expr_repr(expr_dict, comp_type)
        
        # Padd the expression representtaion
        expression_representation.extend([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * (MAX_EXPR_LEN - len(expression_representation)))
        
        # Add the expression representation for this computation to the output
        comps_expr_repr.append(expression_representation)
        
        fused_levels = []
        
        c_code = "C" + str(comp_index)
        # Loop representation for this computation
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
            l_code = c_code + "-L" + str(iter_i)
            
            p_index = comps_placeholders_indices_dict[l_code + "lower_bound"]
            comps_repr[p_index[0]][p_index[1]] = int(program_json["iterators"][iterator_name]["lower_bound"])
            
            p_index = comps_placeholders_indices_dict[l_code + "upper_bound"]
            comps_repr[p_index[0]][p_index[1]] = int(program_json["iterators"][iterator_name]["upper_bound"])
                    
    # Get the index of each feature in the loop representation and replace it with the the information obtained from the schedule
    for loop_name in program_json["iterators"]:
        l_code = "L" + loop_name

        p_index = loops_placeholders_indices_dict[l_code + "upper_bound"]
        loops_repr[p_index[0]][p_index[1]] = int(program_json["iterators"][iterator_name]["upper_bound"])

        p_index = loops_placeholders_indices_dict[l_code + "lower_bound"]
        loops_repr[p_index[0]][p_index[1]] = int(program_json["iterators"][iterator_name]["lower_bound"])
        
    loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr), 0)
    computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr), 0)
    comps_expr_repr = torch.tensor([comps_expr_repr]).float()
    
    # Preparing output vector
    comp_name = ordered_comp_list[0]
    comp_dict = program_json["computations"][comp_name]
    output = []
    if schedule_json:
        comp_schedule_dict = schedule_json[comp_name]
        
        
        
    #     loop interchange
        interchange = [0] * 10
        interchange_keys = {
            "01" : 0,
            "02" : 1,
            "03" : 2,
            "04" : 3,
            "12" : 4,
            "13" : 5,
            "14" : 6,
            "23" : 7,
            "24" : 8,
            "34" : 9,
        }
        skewed = 0
        no_affine_transformations = len(comp_schedule_dict["transformations_list"]) == 0
        if not no_affine_transformations:
            # Make sure we're looking at interchange and only interchange
            for transformation in comp_schedule_dict["transformations_list"]:
                # Check if the transformation is an interchange
                if transformation[0] == 1:
                    i_loop = transformation[1]
                    j_loop = transformation[2]
                    interchange[interchange_keys[f"{i_loop}{j_loop}"]] = 1
                # Check if the transformation is a skewing
                if transformation[0] == 3:
                    skewed = 1
        
        output = output + interchange
        
        # Tiling and parallelization tags
        tiled = 0
        parallelized = 0
        tile_factor_indices = []
        tile_factors = []
        for iter_i, iterator_name in enumerate(comp_dict["iterators"]):
            if comp_schedule_dict["tiling"] and (
                        iterator_name in comp_schedule_dict["tiling"]["tiling_dims"]
                    ):
                tiled = 1
                tile_factor_index = comp_schedule_dict["tiling"]["tiling_dims"].index(
                    iterator_name
                )
                tile_factor_indices.append(tile_factor_index)
                tile_factors.append(int(
                    comp_schedule_dict["tiling"]["tiling_factors"][tile_factor_index])
                )
            if iterator_name == comp_schedule_dict["parallelized_dim"]:
                    parallelized = 1
        output = output + [tiled]
        
    #     output = output + [parallelized, parallelized_loop]
        output = output + [parallelized]
        output = output + [skewed]
        # Unrolling
        unrolled = 0
        if comp_schedule_dict["unrolling_factor"]:
            unrolled = 1
            
    # #     output = output + [unrolled, unroll_factor]
        output = output + [unrolled]
    output_tensor = torch.unsqueeze(torch.FloatTensor(output), 0)
#     print(f"computation tensor shape {computations_tensor.shape}")
#     print(f"loops_tensor tensor shape {loops_tensor.shape}")
    return computations_tensor, loops_tensor, comps_expr_repr, output_tensor

# Get the representation of a specific set of functions and write it to a pkl file for the parent process to read
def get_func_repr_task(input_q, output_q):
    # Recieve functions to work on from parent process
    process_id, programs_dict, repr_pkl_output_folder, train_device = input_q.get()
    function_name_list = list(programs_dict.keys())
    dropped_funcs = []
    local_list = []
    for function_name in tqdm(function_name_list):
        nb_dropped = 0
        # Check whether this function should be dropped
        if drop_program(programs_dict[function_name], function_name):
            dropped_funcs.append(function_name)
            continue
            
        # Get the JSON representation of the program features
        program_json = programs_dict[function_name]["program_annotation"]
        program_schedules = programs_dict[function_name]["schedules_list"]
        tree_structure = program_schedules[0]["tree_structure"]
        # Extract the representation template for the datapoint
        try:
            (
                prog_tree,
                comps_repr_templates_list,
                loops_repr_templates_list,
                comps_placeholders_indices_dict,
                loops_placeholders_indices_dict
            ) = get_representation_template(
                program_json,
                tree_structure,
                train_device=train_device,
            )
        
        except (LoopsDepthException, NbAccessException):
            # If one of the two exceptions was raised, we drop all the schedules for that program and skip to the next program
            nb_dropped =+ len(
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
            "outputs_list": [],
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
                continue
                
            # Calculate the speed up obtained from applying the list of transformations spesified by the schedule
            sched_speedup = program_exec_time / sched_exec_time
                
            # Check whether we can clip the obtained speedup
            sched_speedup = speedup_clip(sched_speedup)
            
            # Fill the obtained template with the corresponsing schedule features
            try:
                comps_tensor, loops_tensor, comps_expr_repr, output_tensor  = get_schedule_representation(
                    program_json,
                    schedule_json,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                )
            except NbTranformationException:
                # If the number of transformations exceeded the specified max, we skip this schedule
                nb_dropped += 1 
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
                comps_expr_repr
            )
            local_function_dict['outputs_list'].append(output_tensor)
        
        # Add the function information to the output list
        local_list.append((
            function_name, 
            nb_dropped, 
            tree_footprint, 
            local_function_dict
        ))
        
    # Write the output representation into pkl files for the parent process to read
    pkl_part_filename = repr_pkl_output_folder + '/pickled_representation_part_'+str(process_id)+'.pkl'
    with open(pkl_part_filename, 'wb') as f:
        pickle.dump(local_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Send the file path to the parent process
    output_q.put((process_id, pkl_part_filename))

# A class to contain the training and validation datasets
# Parameters:
#     dataset_filename: path to training/val dataset
#     max_batch_size: maximum batch size
#     drop_sched_func: function specifying which schedules in the dataset we want to be dropped if any
#     drop_prog_func: function specifying which programs in the dataset we want to be dropped if any
#     speedups_clip_func: function spesifying which speedups to clip and to what value
#     store_device: device where to store the dataset
#     train_device: device where to train the model
class Dataset_parallel:
    def __init__(
        self,
        dataset_filename,
        max_batch_size = 1024,
        drop_sched_func = None,
        drop_prog_func = None,
        speedups_clip_func = None,
        no_batching = False,
        store_device = "cpu",
        train_device = "cpu",
        repr_pkl_output_folder = "none",
        just_load_pickled_repr = False,
        nb_processes = 15
    ):

        # Structure to contain the batched inputs
        self.batched_X = []
        # Structure to contain the batched labels
        self.batched_Y = []
        # Number of all the dropped schedules
        self.nb_dropped = 0
        
        # Number of dropped schedules due to the drop schedule function only
        self.nb_pruned = 0
        # List of dropped functions 
        self.dropped_funcs = []
        # Saved data attributes for analysis and expirements
        self.batched_datapoint_attributes = []
        # Number of loaded datapoints
        self.nb_datapoints = 0
        # number of batches that can fit in the GPU
        self.gpu_fitted_batches_index = -1
        processes_output_list = []
        programs_dict = {}
        batches_dict = dict()
        if just_load_pickled_repr: # Just load the existing repr
            for pkl_part_filename in tqdm(list(Path(repr_pkl_output_folder).iterdir())):
                pkl_part_filename = str(pkl_part_filename)
                with open(pkl_part_filename, 'rb') as f:
                    lst = pickle.load(f)
                processes_output_list.extend(lst)
        else:
            # Separate the function according to the nb_processes parameter
            # Each process will extract the representation for a subset of functions and save that representation into a pkl file
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
                    
                programs_dict = json.loads(dataset_str)
                del dataset_str
                gc.collect()
            elif dataset_filename.endswith("pkl"):
                with open(dataset_filename, "rb") as f:
                    programs_dict = pickle.load(f)
                    
            functions_list = list(programs_dict.keys())
            random.Random(42).shuffle(functions_list)

            nb_funcs_per_process = (len(functions_list)//nb_processes)+1
            print("number of functions per process: ",nb_funcs_per_process)

            for i in range(nb_processes):
                process_programs_dict=dict(list(programs_dict.items())[i*nb_funcs_per_process:(i+1)*nb_funcs_per_process])
                input_queue.put((i, process_programs_dict, repr_pkl_output_folder, store_device))
                
            
            for i in range(nb_processes):
                process_id, pkl_part_filename = output_queue.get()
                # If we want to do batching immediatly after the processes are done, we read the pkl files from the child processes
                if not no_batching:
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
        # If we don't want to proceed to batching, stop here
        # This is used so that the memory for each process if freed before batching
        # To do that we return after the processes are done writing their pkls, then we call instanciate the class again to read the written pkls
        if no_batching:
            print("Parameter no_batching is True. Stopping after the PKL files were saved.")
            return
        
        print("Assembling schedules from each function")
        print(len(processes_output_list))
        for function_name, nb_dropped, tree_footprint, local_function_dict in processes_output_list:
            for node in local_function_dict['tree']["roots"]:
                tree_indices_to_device(node, train_device="cpu")
            batches_dict[tree_footprint] = batches_dict.get(
                tree_footprint, {
                    'tree': local_function_dict['tree'], 
                    'comps_tensor_list': [], 
                    'loops_tensor_list': [ ], 
                    'datapoint_attributes_list': [], 
                    'comps_expr_tree_list': [], 
                    'outputs_list': [], 
                    'exec_time_list': [], 
                    "func_id": []
                }
            )
            batches_dict[tree_footprint]['comps_tensor_list'].extend(local_function_dict['comps_tensor_list'])
            batches_dict[tree_footprint]['loops_tensor_list'].extend(local_function_dict['loops_tensor_list'])
            batches_dict[tree_footprint]['datapoint_attributes_list'].extend(local_function_dict['datapoint_attributes_list'])
            batches_dict[tree_footprint]["comps_expr_tree_list"].extend(local_function_dict["comps_expr_tree_list"])
            batches_dict[tree_footprint]['outputs_list'].extend(local_function_dict['outputs_list'])

            self.nb_dropped += nb_dropped
            self.nb_datapoints += len(local_function_dict['outputs_list'])
        # Delete unused variables
        del processes_output_list
        del programs_dict
        gc.collect()
        
        print("Batching data")
        storing_device = torch.device(store_device)
        # For each tree footprint in the dataset
        # A footprint represents the structure of each function. We batch functions according to their tree footprint so that the forward pass to the model can be correctly executed
        print(len(batches_dict))
        for tree_footprint in tqdm(batches_dict):
            # Shuffling the lists inside each footprint to avoid having batches with very low program diversity
            zipped = list(
                zip(
                    batches_dict[tree_footprint]["datapoint_attributes_list"],
                    batches_dict[tree_footprint]["comps_tensor_list"],
                    batches_dict[tree_footprint]["comps_expr_tree_list"],
                    batches_dict[tree_footprint]["loops_tensor_list"],
                    batches_dict[tree_footprint]["outputs_list"],
                )
            )

            random.shuffle(zipped)
            (
                batches_dict[tree_footprint]["datapoint_attributes_list"],
                batches_dict[tree_footprint]["comps_tensor_list"],
                batches_dict[tree_footprint]["comps_expr_tree_list"],
                batches_dict[tree_footprint]["loops_tensor_list"],
                batches_dict[tree_footprint]["outputs_list"],
            ) = zip(*zipped)
                
            # Split the data into batches of size max_batch_size
            for chunk in range( 0, len(batches_dict[tree_footprint]["outputs_list"]), max_batch_size, ):
                # Check GPU memory in order to avoid out of memory error
                if storing_device.type == "cuda" and ( torch.cuda.memory_allocated(storing_device.index) / torch.cuda.get_device_properties(storing_device.index).total_memory )> 0.83:
                    print( "GPU memory on " + str(storing_device) + " nearly full, switching to CPU memory" )
                    self.gpu_fitted_batches_index = len(self.batched_X)
                    storing_device = torch.device("cpu")
                
                self.batched_datapoint_attributes.append(
                    batches_dict[tree_footprint]["datapoint_attributes_list"][
                        chunk: chunk + max_batch_size
                    ]
                )
                # Append a new batch to the list of batched inputs
                self.batched_X.append(
                    (
                        batches_dict[tree_footprint]["tree"],
                        torch.cat( batches_dict[tree_footprint]["comps_tensor_list"][ chunk : chunk + max_batch_size ], 0).to(storing_device),
                        torch.cat( batches_dict[tree_footprint]["loops_tensor_list"][ chunk : chunk + max_batch_size ], 0).to(storing_device),
                        torch.cat(batches_dict[tree_footprint]["comps_expr_tree_list"][chunk : chunk + max_batch_size],0).to(storing_device),
                    )
                )
                # Append a new batch to the list of batched labels
                self.batched_Y.append(
                    torch.cat( batches_dict[tree_footprint]["outputs_list"][ chunk : chunk + max_batch_size ], 0).to(storing_device)
                )
            # Free up memory since the torch.cat function will allocate a new tensor and copy the content of the parameters
            del batches_dict[tree_footprint]["comps_tensor_list"]
            del batches_dict[tree_footprint]["loops_tensor_list"] 
            del batches_dict[tree_footprint]["comps_expr_tree_list"]
            del batches_dict[tree_footprint]["outputs_list"]
            del batches_dict[tree_footprint]["datapoint_attributes_list"]
        
        # Delete the batches dictionary since it won't be needed anymore
        del batches_dict
        gc.collect()
        
        # Save the size of data that can fit into the GPU. Will be used later when loading the data
        if self.gpu_fitted_batches_index == -1:
            self.gpu_fitted_batches_index = len(self.batched_X)
        
        print( f"Number of datapoints {self.nb_datapoints} Number of batches {len(self.batched_Y)} Number of dropped points from each funtion {self.nb_dropped}")
        
    # Returns batch "index" or range of batches
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            return (self.batched_X[index], self.batched_Y[index])
    
    # Length of the dataset
    def __len__(self):
        return len(self.batched_Y)    

# Function to read the pkls written by the load_data_into_pkls_parallel function, batch the loaded data and return the batched data to be saved
def load_pickled_repr(repr_pkl_output_folder=None,max_batch_size = 1024, store_device="cpu", train_device="cpu"):
    dataset = Dataset_parallel(
        None, 
        max_batch_size, 
        None, 
        repr_pkl_output_folder=repr_pkl_output_folder, 
        just_load_pickled_repr=True, 
        store_device=store_device, 
        train_device=train_device)
    
    indices = list(range(len(dataset)))
    batches_list = []
    for i in indices:
        batches_list.append(dataset[i])

    return dataset, batches_list, indices, dataset.gpu_fitted_batches_index         

# Function to write the representation of the dataset into pkl files. This is done in parallel using the multiprocessing module 
def load_data_into_pkls_parallel(train_val_dataset_file, nb_processes=15, repr_pkl_output_folder=None, overwrite_existing_pkl=False):
    
    if Path(repr_pkl_output_folder).is_dir() and overwrite_existing_pkl:
        shutil.rmtree(repr_pkl_output_folder)
        print('Deleted existing folder ', repr_pkl_output_folder)
        
    Path(repr_pkl_output_folder).mkdir(parents=True, exist_ok=False)
    print('Created folder ', repr_pkl_output_folder)
    
    # Read the JSONs and write the representation into the specified PKL path
    print("Loading data from: "+train_val_dataset_file)
    dataset = Dataset_parallel(
        train_val_dataset_file,
        no_batching=True,
        just_load_pickled_repr=False,
        nb_processes=nb_processes, 
        repr_pkl_output_folder=repr_pkl_output_folder, 
        store_device="cpu", 
        train_device="cpu"
    )         
    return  

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

# A function to extract the transformations applied on a spesific computation in the form of a vector of tags
# Padding is added if the number of transformations is less than the maximum value of MAX_NUM_TRANSFORMATIONS
# The tag representation is as follows:
#         ['type_of_transformation', 'first_interchange_loop', 'second_interchange_loop', 'reversed_loop', 'first_skewing_loop', 'second_skewing_loop', 'third_skewing_loop', 'skew_parameter_1', 'skew_parameter_2', 'skew_parameter_3', 'skew_parameter_4', 'skew_parameter_5', 'skew_parameter_6', 'skew_parameter_7', 'skew_parameter_8', 'skew_parameter_9']
#     Where the type_of_transformation tag is:
#         - 0 for no transformation being applied
#         - 1 for loop interchange
#         - 2 for loop reversal
#         - 3 for loop skewing
# In the case for skewing we are specifying the new values for the transformed submatrix
def get_padded_transformation_tags(
    program_json, schedule_json, comp_name
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
# Add padding to the read/write access matrices
def pad_access_matrix(access_matrix):
    access_matrix = np.array(access_matrix)
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix]
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix]
    padded_access_matrix = np.zeros((MAX_DEPTH + 1, MAX_DEPTH + 2))
    padded_access_matrix[
        : access_matrix.shape[0], : access_matrix.shape[1] - 1
    ] = access_matrix[:, :-1]
    padded_access_matrix[: access_matrix.shape[0], -1] = access_matrix[:, -1]

    return padded_access_matrix

# Tranfrom the access relations to matrices
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
        inputs = (
                    inputs[0],
                    inputs[1].to(train_device),
                    inputs[2].to(train_device),
                    inputs[3].to(train_device),
                    inputs[4].to(train_device),
                    inputs[5].to(train_device),
                )
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
                    inputs[3].to(original_device),
                    inputs[4].to(original_device),
                    inputs[5].to(original_device),
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

# calculates the model scores from the dataframe
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

# Set a lower bound for speedups to avoid fluctuations and make the training easier
def speedup_clip(speedup):
    if speedup < 0.01:
        speedup = 0.01
    return speedup

# Check if this program should be dropped
def drop_program(prog_dict, prog_name):
#     # If there are no schedules explored for this function
#     if len(prog_dict["schedules_list"]) < 2:
#         return True
    
    return False

# Check if this schedule should be dropped
def drop_schedule(prog_dict, schedule_index):
    schedule_json = prog_dict["schedules_list"][schedule_index]
    # If the execution list is empty or it contains incoherent executions 
    if (not schedule_json["execution_times"]) or min(schedule_json["execution_times"]) < 0: 
        return True
    
    return False

# Get the involved computations from a specific node 
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

# Retrieve the iterators that involve this computation from the schedule tree_structure
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

# One-hot encoding for expressions and their datatypes
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
    
# Get the representation of the whole expression recursively
def get_tree_expr_repr(node, comp_type):
        expr_tensor = []
        if node["children"] != []:
            for child_node in node["children"]:
                expr_tensor.extend(get_tree_expr_repr(child_node, comp_type))
        expr_tensor.append(get_expr_repr(node["expr_type"], comp_type))

        return expr_tensor


# A constraint matrix is the set of linear inequalities that describes the iteration domain.
# Example:
# if the iteration domain D is the follwoing
#     {i > 0
#      i < 128
# D =  j > 0
#      j < 32
#      k > 0
#      k < 64}
# The iterator vector is 
# x = [i, 
#      j, 
#      k]
# The constraint matrix A would be:
#     [-1,   0,   0,
#       1,   0,   0,
# A=    0,  -1,   0,
#       0,   1,   0,
#       0,   0,  -1,
#       0,   0,   1]
# The second hand side of the equation b is the vector:
#     b= [0,
#         128,
#         0,
#         32,
#         0,
#         64]
# Since:
#    D = Ax<b
# Get the matrix describing the initial constraints for this program
def get_padded_initial_constrain_matrix(program_json, schedule_json, comp_name):
    
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
            (0, (MAX_DEPTH)*2 - result.shape[0]),
            (0, MAX_DEPTH - result.shape[1]),
        ],
        mode="constant",
        constant_values=0,
    )
    return result

# Returns the shift of the loop in the non-rectangular case or the bound itself if the bound is an integer
def extract_shift_from_bound(bound):
    if is_int(bound):
        return int(bound)
    if "+" not in bound and "-" not in bound:
        return 0
    
    if "+" in bound:
        return int(bound[bound.find("+")+1:])
    if "-" in bound:
        return -int(bound[bound.find("-")+1:])
    
# Returns a vector that represents the right hand sise of teh constraint matrix inequalities
# (The vector b from the previous example)
def get_padded_second_side_of_the_constraint_equations_original(program_json, schedule_json, comp_name):
    iterators_list = program_json["computations"][comp_name]["iterators"]
    result = []
    for it in iterators_list:
        if(is_int(program_json["iterators"][it]["lower_bound"])):
            result.append(int(program_json["iterators"][it]["lower_bound"]))
        else:
#             shift = extract_shift_from_bound(program_json["iterators"][it]["lower_bound"])
            result.append(0)
        if(is_int(program_json["iterators"][it]["upper_bound"])):
            result.append(int(program_json["iterators"][it]["upper_bound"]))
        else:
#             shift = extract_shift_from_bound(program_json["iterators"][it]["upper_bound"])
            result.append(0)
    result = result + [0]*(MAX_DEPTH*2-len(result))
    return result

# In the case where the bound is shifted, remove the shift and return the iterator only
# Example: bound: i + 1 the function return i
def remove_shift_from_bound(bound):
    # If the bound is an integer or it doesn't have a shift there is nothing to be done
    if is_int(bound) or ("+" not in bound and "-" not in bound):
        return bound
    if "+" in bound:
        return bound[:bound.find("+")]
    if "-" in bound:
        return bound[:bound.find("-")]

# Get the matrix describing the iteration domain after applying a sequence of affine transformations
# The transformed constraint matrix is: the original constraint matrix multiplied by the inverse of the transformation matrix
def get_padded_transformed_constrain_matrix(program_json, schedule_json, comp_name):
    iterators_list = program_json["computations"][comp_name]["iterators"]
    
    # Extract the transformations matrix for this schedule
    transformation_matrix = get_transformation_matrix(program_json, schedule_json, comp_name)
    
    # Create the initial constraint matrix without any padding
    result = []
    for i in iterators_list:
        for j in range(2):
            if j == 0 :
                result.append(format_bound(i, program_json["iterators"][i]["lower_bound"], iterators_list, True))
            else:
                result.append(format_bound(i, program_json["iterators"][i]["upper_bound"], iterators_list, False))
    # Get the inverse of the transformation matrix
    inverse = np.linalg.inv(transformation_matrix)
    
    # Multiply thw two to gte the transformed constraint matrix
    result = np.matmul(result, inverse)
    result = np.array(result)
    
    # Add padding
    result = np.pad(
        result,
        [
            (0, (MAX_DEPTH)*2 - result.shape[0]),
            (0, MAX_DEPTH - result.shape[1]),
        ],
        mode="constant",
        constant_values=0,
    )
    
    return result

# Check whether the string contains an integer and return true if so
def is_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

# Helper function to return lines from the constraint matrix
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

# Convert a tags vector describing an affine transfromation (Reversal, Skewing, Interchange) into a matrix that represents the same transformation
def get_trasnformation_matrix_from_vector(transformation, matrix_size):
    matrix = np.identity(matrix_size)
    assert(len(transformation) == MAX_TAGS)
    if (transformation[0] == 1):
        # Interchange
        assert(transformation[1] < matrix_size and transformation[2] < matrix_size)
        matrix[transformation[1], transformation[2]] = 1
        matrix[transformation[1], transformation[1]] = 0
        matrix[transformation[2], transformation[1]] = 1
        matrix[transformation[2], transformation[2]] = 0

    elif (transformation[0] == 2):
        # Reversal
        assert(transformation[3] < matrix_size)
        matrix[transformation[3], transformation[3]] = -1

    elif transformation[0] == 3:
        # 2D Skewing
        if transformation[6] == 0:
            
            assert(transformation[4] < matrix_size and transformation[5] < matrix_size)
            matrix[transformation[4], transformation[4]] = transformation[7]
            matrix[transformation[4], transformation[5]] = transformation[8]
            matrix[transformation[5], transformation[4]] = transformation[9]
            matrix[transformation[5], transformation[5]] = transformation[10]
        if transformation[6] > 0:
            # 3D skeweing
            assert(transformation[4] < matrix_size and transformation[5] < matrix_size and transformation[6] < matrix_size)
            matrix[transformation[4], transformation[4]] = transformation[7]
            matrix[transformation[4], transformation[5]] = transformation[8]
            matrix[transformation[4], transformation[6]] = transformation[9]
            matrix[transformation[5], transformation[4]] = transformation[10]
            matrix[transformation[5], transformation[5]] = transformation[11]
            matrix[transformation[5], transformation[6]] = transformation[12]
            matrix[transformation[6], transformation[4]] = transformation[13]
            matrix[transformation[6], transformation[5]] = transformation[14]
            matrix[transformation[6], transformation[6]] = transformation[15]
        
    return matrix

# Transform a sequence of transformation vectors into a single transfromation matrix that represents the whole sequence
def get_transformation_matrix(
    program_json, schedule_json, comp_name
):
    nb_iterators = len(program_json["computations"][comp_name]["iterators"])
    final_transformation = np.identity(nb_iterators)
    for transformation in schedule_json[comp_name]["transformations_list"]:
        matrix = get_trasnformation_matrix_from_vector(transformation, nb_iterators)
        final_transformation = np.matmul(matrix, final_transformation)
    return final_transformation

# Returns a string representation of a schedule and the transformations applied in it
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
                # if this computation was involved in a fusion, we know it uses the same iterators as the computation it was fused with
                if name in fusion:
                    iterator_comp_name = fusion[0]
                    transf_loop_nest = program_json["computations"][iterator_comp_name]["iterators"].copy()
                    schedule = sched_json[iterator_comp_name]
                
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

def tree_indices_to_device(node, train_device):
    node['loop_index'] = node['loop_index'].to(train_device, non_blocking=True)
    if 'computations_indices' in node:
        node['computations_indices'] = node['computations_indices'].to(
            train_device, non_blocking=True)
    for child in node['child_list']:
        tree_indices_to_device(child, train_device)    