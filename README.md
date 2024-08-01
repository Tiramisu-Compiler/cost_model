# Tiramisu Cost Model
A deep learning model to predict the speedup obtained from applying a sequence of transformations on an input program.

## Installation  
Install the environment using the `environment.yml` file as follows:
```bash
conda env create -f environment.yml
```  
This should create an environment called `cost_model_env`.

Whenever you want to use the model, you need to activate the environment as follows:
```bash
conda activate cost_model_env
```  

## Configuring the repository
All of the main scripts use Hydra for configuration management. To configure the repository, fill the configuration template `conf/config.yaml` with the paths and parameters required.
While using one of the following script files, you can override any configuration in the conf file. For example, to modify the batch size to 512 for training, use the following command. The parameter should be included with its section name.  
```
python generate_dataset.py data_generation.batch_size=512
```

## Generating the dataset
Currently, we have separated the data loading and training from each other. This is because the data loading is very time-consuming, and we don't want to redo it for every training. To solve this, we run a script to load the raw data (JSON), extract the representation for each datapoint, and then save the batched data in a `.pt` file that can be loaded directly into memory for training. We call this process data generation.
To generate the dataset, run the python script `generate_dataset.py` (after configuring the repository):  
```bash
python generate_dataset.py
```

## Training the model
To run the training, run the python script `train_model.py` (after configuring the repository and generating the dataset):  
```bash
python train_model.py
```

## Using wandb for visualization
The repository allows the use Weights and Biases for visualization. To enable it, set the `use_wandb` parameter to `True`, after logging into wandb from the command line. The project name should be specified. This name does not have to already exist in wandb. During training, the progress can be found on the [wandb platform](https://wandb.ai/). 

## Evaluation of the trained model
To evaluate the trained model, run the python script `evaluate_model.py` (after configuring the repository and generating the dataset):  
```bash
python evaluate_model.py
```
