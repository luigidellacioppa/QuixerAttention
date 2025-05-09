# Quixer
## A Quantum Transformer Model

### Installation
```
pip install -r requirements.txt
```
## Running all models on a local machine

On CPU:
```
python ./run_comprehensive.py cpu
```

On Nvidia GPU:
```
python ./run_comprehensive.py cuda
```

## Running a single model on a cluster with Slurm

### Preparing the environment

```
ssh adellacioppa@unisa.it@hnode.unisa.it
```

Load all the necessary modules:
```
module load python3
module load cuda11.7/blas/11.7.1
module load cuda11.7/toolkit/11.7.1
```

### Create a virtual environment and install the necessary packages
```
conda create -n Quixer python=3.10
conda activate Quixer
pip install -r requirements.txt
```

### Running the model
To run the model on the cluster, we can:
- execute the command:
```
srun [options] python run_comprehensive.py cuda
```
where options are the parameters for the Slurm scheduler:
```
--job-name=Quixer
--partition=gpuq #
--nodes=1
--ntasks=1
--cpus-per-task=1
--gres=gpu:1
--nodelist=gnode03 # or any other free node (to get the list of available nodes, run the command: sinfo)
-o slurm_Quixer.out
-e slurm_Quixer.err
```
- create a script that will be executed by the Slurm scheduler. 
The script is called `Quixer.sh` and is located in the root directory of the project:
```
#!/bin/bash
#SBATCH --job-name=Quixer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpuq
#SBATCH --nodelist=gnode03
#SBATCH --gpus-per-task=1
#SBATCH -o slurm_Quixer.out
#SBATCH -e slurm_Quixer.err

srun python ./run_comprehensive.py cuda
```
To run the script, check which node in the partition is free, then change the 
node in the nodelist option (`gnode0n`) and execute the following command:
```
sbatch Quixer.sh
```

### Cheching job and job step information for jobs managed by Slurm
To check the job information for jobs managed by Slurm, we can use the following command:
```
squeue 
```

### Monitoring the job
To monitor the job, we can use the following commands:
```
squeue -u adellacioppa@unisa.it
squeue -j <job_id>
```

### Checking the status of the nodes
To check the status of the nodes, we can use the following command:
```
sinfo
```

### Checking the nodes with free resources
To check the free nodes, we can use the following command:
```
sinfo -o "%N %t"
```

### Checking the resources of a node
To check the resources of a node, we can use the following command:
```
scontrol show node <node_name>
```

### Check the status of all the nodes
To check the status of all the nodes, we can use the following command:
```
scontrol show nodes
```

### Stopping the job
To stop the job, we can use the following command:
```
scancel <job_id>
```

### Checking the output
The output of the job is saved in the files `slurm_Quixer.out` and `slurm_Quixer.err`.
The first file contains the output of the program, while the second file contains the error messages.

