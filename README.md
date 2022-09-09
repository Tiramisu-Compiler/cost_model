# resnet_cost_model
Train a recursive lstm architecture on Tiramisu schedules.


## Installation  
Here are the steps of the installtion:  
1. Install pytorch. Note that the repo was tested only with the version 1.10.2 of pytorch.
2. Install the required packages using the command:  
```python
pip install -r requirements.txt
```  


## Configuring the repository
To configure the repository, copy the `conf/conf.yml.example` file to `conf/conf.yml` as follows:  
```bash
# After navigating to the root directory of this repo
cp conf/conf.yml.example conf/conf.yml
```
Modify the configuration according to your personal need.

## Generating the dataset
To generate the dataset, run the python script `generate_dataset.py` (after configuring the repository):  
```bash
python generate_dataset.py
```

## Training the model
To run the training, run the bash script `run.sh` with the GPU number to run the training on (after configuring the repository and generatoing the dataset):  
```bash
bash run.sh [num] # replace [num] with a GPU number
```

## Evaluation of the trained model
To evaluate the trained model, run the python script `evaluate_model.py` (after configuring the repository and generatoing the dataset):  
```bash
python evaluate_model.py
```