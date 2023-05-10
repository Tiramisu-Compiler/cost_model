import hydra
from hydra.core.config_store import ConfigStore
import re
from pathlib import Path
from utils.data_utils import *
from utils.modeling import *
# Code to run cpp files
import subprocess
import os
import json
import logging
os.environ["CXX"] = "c++"
os.environ["CC"] = "gcc"
os.environ['GXX'] = 'g++'

os.environ["TIRAMISU_ROOT"] = "/data/scratch/mmerouani/tiramisu4/tiramisu"
os.environ["LD_LIBRARY_PATH"] = "$TIRAMISU_ROOT/3rdParty/Halide/lib/:$TIRAMISU_ROOT/3rdParty/llvm/build/lib:$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/isl/build/lib"

class ScheduleExecutionCrashed(Exception):
    "Raised when the execution of the schedule crashes"
    pass
def run_cpp_code(cpp_code: str, output_path: str):
    # if Config.config.tiramisu.is_new_tiramisu:
        # Making the tiramisu root path explicit to the env
#     shell_script = [
#         # Compile intermidiate tiramisu file
#         "$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/install/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++17 -O0 -o {}.o -c -x c++ -"
#         .format(output_path),
#         # Link generated file with executer
#         "$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++17 -O0 {}.o -o {}.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/install/lib64  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/install/lib64:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl"
#         .format(output_path, output_path),
#         # Run the program
#         "{}.out".format(output_path),
#         # Clean generated files
#         "rm {}*".format(output_path)
#     ]
    # else:
    shell_script = [
        # Compile intermidiate tiramisu file
        "$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o {}.o -c -x c++ -"
        .format(output_path),
        # Link generated file with executer
        "$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 {}.o -o {}.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/lib  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/lib:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl"
        .format(output_path, output_path),
        # Run the program
        "{}.out".format(output_path),
#         # Clean generated files
#         "rm {}*".format(output_path)
    ]
    try:
        compiler = subprocess.run(["\n".join(shell_script)],
                                    input=cpp_code,
                                    capture_output=True,
                                    text=True,
                                    shell=True,
                                    check=True)
        return compiler.stdout if compiler.stdout != '' else "0"
    except subprocess.CalledProcessError as e:
        print("Process terminated with error code", e.returncode)
        print("Error output:", e.stderr)
        return "0"
    except Exception as e:
        print(e)
        return "0"

# Extract model output from function annotations
def get_function_transformations(model, program_dict):
    
    # Extract the input for the model
    (
        prog_tree,
        comps_repr_templates_list,
        loops_repr_templates_list,
        comps_placeholders_indices_dict,
        loops_placeholders_indices_dict
    ) = get_representation_template(
        program_dict,
        train_device="cpu",
    )
    schedule_json = program_dict['schedules_list'][0]
    program_json = program_dict["program_annotation"]
    comps_tensor, loops_tensor, comps_expr_repr, _  = get_schedule_representation(
                    program_json,
                    schedule_json,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                )
    inputs = (
                prog_tree,
                comps_tensor.to("cpu"),
                loops_tensor.to("cpu"),
                comps_expr_repr.to("cpu"),
            ) 
    # Forward pass
    model.eval()
    model.to("cpu")
    output = model(inputs)
    return output
# Function to read the original code
class TiramisuProgram():
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.annotations = None
        self.comps = None
        self.name = None
        self.schedules_legality = {}
        self.schedules_solver = {}
        self.original_str = None
        self.nb_iterators = 0
        self.wrappers = None
        if (file_path):
            self.load_code_lines()

    def load_code_lines(self, original_str: str = None):
        '''
        This function loads the file code , it is necessary to generate legality check code and annotations
        '''
        if (self.name):
            # if self.name is None the program doesn't exist in the offline dataset but built from compiling
            # if self.name has a value than it is fetched from the dataset, we need the full path to read
            # the lines of the real function to execute legality code
            func_name = self.name
            file_name = func_name + "_generator.cpp"
            file_path = cpps_path + func_name + "/" + file_name
            self.file_path = file_path
        else:
            file_path = self.file_path

        if original_str:
            self.original_str = original_str
        else:
            with open(file_path, 'r') as f:
                self.original_str = f.read()

        self.func_folder = ('/'.join(Path(file_path).parts[:-1])
                            if len(Path(file_path).parts) > 1 else '.') + '/'
        self.body = re.findall(r'(tiramisu::init(?s:.)+)tiramisu::codegen',
                               self.original_str)[0]
        self.name = re.findall(r'tiramisu::init\(\"(\w+)\"\);',
                               self.original_str)[0]
        # Remove the wrapper include from the original string
        self.wrapper_str = f'#include "{self.name}_wrapper.h"'
        self.original_str = self.original_str.replace(
            self.wrapper_str, f"// {self.wrapper_str}")
        self.comps = re.findall(r'computation (\w+)\(', self.original_str)
        self.code_gen_line = re.findall(r'tiramisu::codegen\({.+;',
                                        self.original_str)[0]
        # buffers_vect = re.findall(r'{(.+)}', self.code_gen_line)[0]
        # self.IO_buffer_names = re.findall(r'\w+', buffers_vect)
        # self.buffer_sizes = []
        # for buf_name in self.IO_buffer_names:
        #     sizes_vect = re.findall(r'buffer ' + buf_name + '.*{(.*)}',
        #                             self.original_str)[0]
        #     self.buffer_sizes.append(re.findall(r'\d+', sizes_vect))

# Function to add legality for each transformation
class Optimization():
    def __init__(self, action_type, loop, factor, computation_names):
        self.type = action_type
        self.l0 = loop
        self.factor = factor
        self.comps = computation_names
        self.tiramisu_optim_str = self.get_tiramisu_optim_str()
        
    def get_tiramisu_optim_str(self):
            """Convert the optimization command into Tiramisu code.
            Returns:
                str: The tiramisu snippet that represents the optimization command.
            """
            if self.type == 0:
                return ("\n\t" + self.comps[0] + ".tag_parallel_level(" +
                        str(self.l0) + ");")

            elif self.type == 1:
                tiling_str = ".tile(" + ",".join(
                    [str(p) for p in [self.l0, self.l0+1, self.factor, self.factor]]) + ");"
                optim_str = ""
                for comp in self.comps:
                    optim_str += "\n\t{}".format(comp) + tiling_str
                return optim_str
            elif self.type == 2:
                optim_str = ""
                for comp in self.comps:
                    unrolling_str = (
                        ".unroll(" +
                        ",".join([str(p) for p in [self.l0, self.factor]]) + ");")
                    optim_str += "\n\t{}".format(comp) + unrolling_str
                return optim_str
def get_legality_code(tiramisu_program, optimizations_list, nb_iterators):
    comps = tiramisu_program.comps
    first_comp = comps[0]
    # Add code to the original file to get legality result
    legality_check_lines = '''\n\tprepare_schedules_for_legality_checks();\n\tperforme_full_dependency_analysis();\n\tbool is_legal=true;'''
    for optim in optimizations_list:
        if optim.type == 0:
            legality_check_lines += '''\n\tis_legal &= loop_parallelization_is_legal(''' + str(
                optim.l0
            ) + ''', {&''' + first_comp + '''});\n'''
        elif optim.type == 2:
            level = nb_iterators - 1
            legality_check_lines += '''\n\tis_legal &= loop_unrolling_is_legal(''' + str(
                level) + ''', {''' + ", ".join(
                    [f"&{comp}" for comp in comps]) + '''});'''
        legality_check_lines += optim.tiramisu_optim_str + '\n'

    legality_check_lines += '''
        is_legal &= check_legality_of_function();   
        std::cout << is_legal;
        '''
    # Paste the lines responsable of checking legality of schedule in the cpp file
    cpp_code = tiramisu_program.original_str.replace(
        tiramisu_program.code_gen_line, legality_check_lines)
    return cpp_code
def get_first_legal_parallelization(program, nb_iterators, output_path):
    for i in range(nb_iterators):
        optimization_list = [Optimization(0, i, -1, program.comps)]
        code = get_legality_code(program, optimization_list, nb_iterators)
        output_path = os.path.join(
            output_path, f'{program.name}_legality')
        legal = run_cpp_code(cpp_code=code, output_path=output_path)
        assert( int(legal) == 1 or int(legal) == 0)
        if legal:
            return i
    return None
def get_first_legal_tiling(program, nb_iterators, output_path):
    for i in range(nb_iterators):
        optimization_list = [Optimization(1, i, 32, program.comps)]
        code = get_legality_code(program, optimization_list, nb_iterators)
        output_path = os.path.join(
            output_path, f'{program.name}_legality')
        legal = run_cpp_code(cpp_code=code, output_path=output_path)
        assert( int(legal) == 1 or int(legal) == 0)
        if legal:
            return i
    return None
def get_first_legal_unrolling(program, nb_iterators, output_path):
    for i in range(nb_iterators):
        optimization_list = [Optimization(2, i, 4, program.comps)]
        code = get_legality_code(program, optimization_list, nb_iterators)
        output_path = os.path.join(
            output_path, f'{program.name}_legality')
        legal = run_cpp_code(cpp_code=code, output_path=output_path)
        assert( int(legal) == 1 or int(legal) == 0)
        if legal:
            return i
    return None
# Function to run a generator and return its output
def get_transformed_code(tiramisu_program, optimizations_list):
    comps = tiramisu_program.comps
    nb_iterators = tiramisu_program.nb_iterators
    first_comp = comps[0]
    # Add code to the original file to get legality result
    legality_check_lines = '''\n\tprepare_schedules_for_legality_checks();\n\tperforme_full_dependency_analysis();\n\tbool is_legal=true;'''
    for optim in optimizations_list:
        if optim.type == 0:
            legality_check_lines += '''\n\tis_legal &= loop_parallelization_is_legal(''' + str(
                optim.l0
            ) + ''', {&''' + first_comp + '''});\n'''
        elif optim.type == 2:
            level = nb_iterators - 1
            legality_check_lines += '''\n\tis_legal &= loop_unrolling_is_legal(''' + str(
                level) + ''', {''' + ", ".join(
                    [f"&{comp}" for comp in comps]) + '''});'''
        legality_check_lines += optim.tiramisu_optim_str + '\n'

    legality_check_lines += '''
        is_legal &= check_legality_of_function();   
        assert(is_legal == 1);
        '''
    legality_check_lines += '\n\t' + tiramisu_program.code_gen_line + '\n'
    # Paste the lines responsable of checking legality of schedule in the cpp file
    cpp_code = tiramisu_program.original_str.replace(
        tiramisu_program.code_gen_line, legality_check_lines)
    return cpp_code
def write_to_disk(cpp_code: str, output_path: str, extension: str = '.cpp'):
        with open(output_path + extension, 'w') as f:
            f.write(cpp_code)
            
def get_cpu_exec_times(tiramisu_program, optims_list, function_root_path):
        # Get the code of the schedule
        cpp_code = get_transformed_code(tiramisu_program, optims_list)
        os.environ["TIRAMISU_ROOT"] = "/data/scratch/mmerouani/tiramisu4/tiramisu"
        # Write the code to a file
        output_path = os.path.join(
            function_root_path, tiramisu_program.name)

        cpp_file_path = output_path + '_schedule.cpp'
        write_to_disk(cpp_code, output_path + '_schedule')

        if True:
            # Making the tiramisu root path explicit to the env
            shell_script = [
                f"export TIRAMISU_ROOT=/data/scratch/mmerouani/tiramisu4/tiramisu",
                f"cd {function_root_path}",
                # Compile intermidiate tiramisu file
                f"$CXX -I$TIRAMISU_ROOT/3rdParty/Halide/include -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o {tiramisu_program.name}.o -c {cpp_file_path}",
                # Link generated file with executer
                f"$CXX -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O0 {tiramisu_program.name}.o -o {tiramisu_program.name}.out   -L$TIRAMISU_ROOT/build  -L$TIRAMISU_ROOT/3rdParty/Halide/lib  -L$TIRAMISU_ROOT/3rdParty/isl/build/lib  -Wl,-rpath,$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/Halide/lib:$TIRAMISU_ROOT/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl",
                # Run the generator
                f"./{tiramisu_program.name}.out",
                # compile the wrapper
                f"$CXX -shared -o {tiramisu_program.name}.o.so {tiramisu_program.name}.o",
                f"$CXX -std=c++11 -fno-rtti -I$TIRAMISU_ROOT/include -I$TIRAMISU_ROOT//3rdParty/Halide/include -I$TIRAMISU_ROOT/3rdParty/isl/include/ -I$TIRAMISU_ROOT/benchmarks -L$TIRAMISU_ROOT/build -L$TIRAMISU_ROOT/3rdParty/Halide/lib -L$TIRAMISU_ROOT/3rdParty/isl/build/lib -o {tiramisu_program.name}_wrapper -ltiramisu -lHalide -ldl -lpthread -lm -Wl,-rpath,$TIRAMISU_ROOT/build {tiramisu_program.name}_wrapper.cpp ./{tiramisu_program.name}.o.so -ltiramisu -lHalide -ldl -lpthread -lm -lisl",
            ]

        run_script = [
#             f"ssh lanka16",
            # cd to the workspace
            f"cd {function_root_path}",
            f"export LD_LIBRARY_PATH=$TIRAMISU_ROOT/3rdParty/Halide/lib/:$TIRAMISU_ROOT/3rdParty/llvm/build/lib:$TIRAMISU_ROOT/build:$TIRAMISU_ROOT/3rdParty/isl/build/lib",
            #  set the env variables
            f"export DYNAMIC_RUNS=0",
            f"export MAX_RUNS=30",

            # run the wrapper
            f"./{tiramisu_program.name}_wrapper",
        ]
        try:
            # run the compilation of the generator and wrapper
            compiler = subprocess.run([" ; ".join(shell_script)],
                                      capture_output=True,
                                      text=True,
                                      shell=True,
                                      check=True)
            # run the wrapper and get the execution times
            compiler = subprocess.run([" ; ".join(run_script)],
                                      capture_output=True,
                                      text=True,
                                      shell=True,
                                      check=True)

            # Extract the execution times from the output and return the minimum
            if compiler.stdout:
                results = [float(x) for x in compiler.stdout.split()]
                return results
            else:
                logging.error("No output from schedule execution")
                logging.error(compiler.stderr)
                logging.error(compiler.stdout)
                logging.error(
                    f"The following schedule execution crashed: {tiramisu_program.name}, schedule: {optims_list} \n\n {cpp_code}\n\n")
                raise ScheduleExecutionCrashed(
                    "No output from schedule execution")
        except subprocess.CalledProcessError as e:
            print("Process terminated with error code", e.returncode)
            print("Error output:", e.stderr)
            print("Output:", e.stdout)
            raise ScheduleExecutionCrashed(
                f"Schedule execution crashed: function: {tiramisu_program.name}, schedule: {optims_list}")
        except Exception as e:
            print(e)
            raise e

# Function to applly a transformation and measure its execution time
def compile_initial_schedule(tiramisu_program, output_path):
    # TODO : add getting tree structure object from executing the file instead of building it
    output_path = os.path.join(
        output_path, f'{tiramisu_program.name}')
    # Add code to the original file to get json annotations

    # if Config.config.tiramisu.is_new_tiramisu:
    get_json_lines = '''
        auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function(), {});
        std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_schedule_json(ast);
        std::cout << program_json;
        '''
    # else:
    #     get_json_lines = '''
    #         auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function());
    #         std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_program_json(ast);
    #         std::cout << program_json;
    #         '''
    # Paste the lines responsable of generating the program json tree in the cpp file
    cpp_code = tiramisu_program.original_str.replace(
        tiramisu_program.code_gen_line, get_json_lines)
    return run_cpp_code(cpp_code=cpp_code, output_path=output_path)
# Function to applly a transformation and measure its execution time
def compile_annotations(tiramisu_program, output_path):
    # TODO : add getting tree structure object from executing the file instead of building it
    output_path = os.path.join(
        output_path, f'{tiramisu_program.name}')
    # Add code to the original file to get json annotations

    # if Config.config.tiramisu.is_new_tiramisu:
    get_json_lines = '''
        auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function(), {});
        std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_program_json(ast);
        std::cout << program_json;
        '''
    # else:
    #     get_json_lines = '''
    #         auto ast = tiramisu::auto_scheduler::syntax_tree(tiramisu::global::get_implicit_function());
    #         std::string program_json = tiramisu::auto_scheduler::evaluate_by_learning_model::get_program_json(ast);
    #         std::cout << program_json;
    #         '''
    # Paste the lines responsable of generating the program json tree in the cpp file
    cpp_code = tiramisu_program.original_str.replace(
        tiramisu_program.code_gen_line, get_json_lines)
    return run_cpp_code(cpp_code=cpp_code, output_path=output_path)
def load_model(conf, path):
    model = Model_Recursive_LSTM_v2(
        input_size=conf.model.input_size,
        comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
        drops=list(conf.model.drops),
        loops_tensor_size=2,
        device="cpu",
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
def transform_function(model, workspace, function_name, output_file_path):
    
    function_path = os.path.join(
        workspace, function_name)
    cpps_path = f"{function_path}/{function_name}_generator.cpp"
    prog = TiramisuProgram(cpps_path)
    program_annotations = json.loads(compile_annotations(prog, function_path))
    first_schedule = json.loads(compile_initial_schedule(prog, function_path))
    nb_iterators = len(program_annotations["iterators"])
    prog.nb_iterators = nb_iterators
    program_dict = {}
    program_dict["program_annotation"] = program_annotations
    program_dict["schedules_list"] = []
    program_dict["schedules_list"].append(first_schedule)
    output = get_function_transformations(model, program_dict)
    threshold = .5
    out = [1 if output[0][i]>threshold else 0 for i in range(output.shape[1])]
    
    tiling_level = parallelization_level = unrolling_level = None
    if out[0]:
        # Tiling
        tiling_level = get_first_legal_tiling(prog, nb_iterators, function_path)

    if out[1]:
        # Parallelization
        parallelization_level = get_first_legal_parallelization(prog, nb_iterators, function_path)
            
    if out[2]:
        # Unrollling
        unrolling_level = get_first_legal_unrolling(prog, nb_iterators, function_path)

    optimzation_list = []
#     if parallelization_level is not None:
#         optimzation_list.append(Optimization(0, parallelization_level, -1, prog.comps))
#     if tiling_level is not None:
#         optimzation_list.append(Optimization(1, tiling_level, 32, prog.comps))
#     if unrolling_level is not None:
#         optimzation_list.append(Optimization(2, unrolling_level, 4, prog.comps))
        
    
    code = get_transformed_code(prog, optimzation_list)
    results = get_cpu_exec_times(prog, optimzation_list, function_path)
    execution_time = min(results)
    with open(output_file_path, "a") as f:
        f.write(f"{prog.name}, {execution_time}, {out}\n")
        
@hydra.main(config_path="conf", config_name="config")        
def main(conf):
    model = load_model(conf, "/data/commit/tiramisu/data_factory_kb4083/code2sched/cost_model/best_model_Code2sched_PTU_modified_output_loss_no_softmax_div_4_20000_c6a.pt")
    print("Done loading model")
    workspace = "/data/commit/tiramisu/data_factory_kb4083/code2sched/cost_model/progs"
    functions_list = [directory for directory in next(os.walk(workspace))[1] if "function" in directory]
    for function_name in functions_list:
        print(f"Working on {function_name}")
        transform_function(model, workspace, function_name, "/data/commit/tiramisu/data_factory_kb4083/code2sched/cost_model/progs/execution_times.txt")
        print(f"{function_name} done")
if __name__ == "__main__":
    main()