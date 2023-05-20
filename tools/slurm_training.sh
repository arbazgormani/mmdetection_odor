#!/bin/bash -l
#SBATCH --cpus-per-task=8
#SBATCH --job-name=Model-training
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:4
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

echo "Into extracting data"
#start Extracting the dataset
python diffusion/extract/extract.py

GPUS=4
CONFIG=$1
WORK_DIR='/home/hpc/iwi5/iwi5108h/workspace/stable-diffusion-data-augmentation/mmdetection/work_dirs/odor_config'

echo 'train with ' ${CONFIG}

#start training
./tools/dist_train.sh ${CONFIG} ${GPUS}

echo "train done"
