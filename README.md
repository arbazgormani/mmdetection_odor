# Image in-painting as Augmentation

This repo can be used to train the model on synthetic images of ODOR Dataset.

## Requirements
1) Python
2) Pytorch
3) MM Detection Framework Installation
4) Stable Diffusion Model


## Installation

1) Use the Anaconda package manager [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) to install python.
2) Install Pytorch using the following command.

```bash
pip3 install torch torchvision torchaudio
```
3) Clone the repository and Install mmDetection framework using this follwoing commands.
```bash
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
mim install mmdet
```
4) Install the [Stable Diffusion In-painting model](https://huggingface.co/runwayml/stable-diffusion-inpainting).
## How to run

```bash
# 1) mkdir datasets
# 2) paste your dataset in this directory and change the path in config file.

cd diffusion/
python main.py  # To generate the synthetic data using Stable diffusion model.

# To train the model
# make sure you are in mmDetection directory and run
pyhton tools/train.py configs/odor/odor_config.py

# To train the model using slurm
sbatch slurm_training.sh
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Author

Muhammad Arbaz
