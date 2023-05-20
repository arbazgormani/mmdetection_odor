#!/bin/bash -l
#SBATCH --cpus-per-task=8
#SBATCH --job-name=Stable-diffusion
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
# do not export environment variables 
#SBATCH --export=NONE 

# do not export environment variables 
unset SLURM_EXPORT_ENV 
echo "Job is running on" ${hostname}

#enable proxy
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

#activate conda environment
conda activate pytorch


#start generating
python diffusion/main.py
echo 'yay'
