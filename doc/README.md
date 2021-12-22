# EPFL ML Project 2 - Road Segmentation

Implementation by team CTM. 

## :mag_right: Abstract
This repository contains the code for the Road Segmentation challenge on AICrowd. 
This challenge is a case of binary semantic segmentation. We must first predict the pixel-wise class 
(background or road) and then predict the class of each 16x16 patch from a test image, by thresholding on the average number of road pixels. 

Key features of our approach:
* data augmentations
* UNet architecture with bilinear upsampling layers
in the upsampling branch and a pretrained VGG13 network in the downsampling branch
* added dropout after each upsampling or downsampling block
* split test image in 4 patches of 400x400 pixels and average predictions to get 608x608 original test image
* test time augmentations

## :wrench: Environment setup
For our experiments we used PyCharm IDE.
Tested configurations: 

* `Python 3.7`
* `pytorch 1.1.0`
* `torchvision 0.3.0`
* `CUDA 9.0`
* `CUDNN 7.0.5 `

All experiments were run on an `NVIDIA GTX 950M` (4 GB RAM).

### 1.Libraries installation
*NOTE*: You should have [Anaconda](https://docs.anaconda.com/anaconda/install/windows/) installed.

For PyTorch and torchvision with CUDA installation use the following command:
```bash
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
```
*NOTE*: Check your CUDA + CuDNN version and PyTorch compatibility.

For PyTorch and torchvision with cpu only (NOT RECOMMENDED) installation use the following command:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

For the rest of the libraries use the next command:
```bash
pip install numpy imutils matplotlib opencv-python Pillow PyYAML tensorboard future tqdm ttach
```

### 2.Set PYTHONPATH
Please add the project root folder to `$PYTHONPATH` using following command:
```bash
export PYTHONPATH=$PYTHONPATH:<path_to_project_folder>
```
 
## :file_folder: Data and Folders Structure
Please download the data at this [link](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files?unique_download_uri=2869&challenge_id=68).
Unzip the `training.zip` and `test_set_images.zip` under the `data` folder.

Targeted folder structure:
```
└── PROJECT_ROOT
       ├── data                 <- dataset
       |   ├── training
       |   |   ├──  groundtruth
       |   |   └──  images
       |   └── test_set_images
       |       ├── test_1
       |       ├── test_2
       |       └── ...
       ├── checkpoints          <- models weights    
       ├── configs              <- configuration files
       ├── doc                  <- readme and images
       ├── logs                 <- experiments log files
       ├── pretrained_weights   <- pretrained weights
       ├── src                  <- train, predict and data augmentation scripts
       ├── utils                <- multiple utility scripts grouped by functionality
       ├── results              <- submission files
       └── visualizations       <- predictions output folder

```

## :brain: Pretrained weights
We use VGG13 pretrained on ImageNet weights. You can download them at [this](https://download.pytorch.org/models/vgg13_bn-abd245e5.pth) link.
Please place them under `pretrained_weights` folder. 

## :muscle: Data augmentations
We use offline data augmentation. Therefore, before starting the training process,
you must run the following command in order to augment the data:
```bash
cd src
python data_augmentation.py <augmentation_config_filename>
```
The parameter given, `augmentation_config_filename`, refers to the augmentation configuration file 
used for generating the new data. In order to obtain the BEST RESULTS, use `augment_3`:
```bash
cd src
python data_augmentation.py augment_3
```
In this way, the folder structure will be changed in the following way:
```
└── PROJECT_ROOT
       ├── data  
       |   ├── training
       |   |   ├──  groundtruth
       |   |   ├──  groundtruth_augment_3
       |   |   ├──  images
       |   |   └──  images_augment_3
       |   └── test_set_images
       |       ├── test_1
       |       ├── test_2
       |       └── ...
       ...
```

## :scroll: Configs
Check `config` folder for different configs for data augmentation and training/val/predict of the network.
Configs follow a YAML format. We used them in order to keep track of our experiments with different parameters.
 
A `augment_*` configuration files are used for data augmentation. When creating an augmentation config file,
set these 2 parameters in this way:
```
train_data: "data/training/images_<augmentation_config_filename>/"
gt_data: "data/training/groundtruth_<augmentation_config_filename>/"
```

A `experiment_*` configuration files are used for training/val/predict.

When creating an experiment config file, set these 3 parameters in this way:
```
augmentation_type: '<augmentation_config_filename>'
train_data: "data/training/images_<augmentation_config_filename>/"
gt_data: "data/training/groundtruth_<augmentation_config_filename>/"
```

## :running: Running experiments
### Training and validation

For training and validation run the following commands:
```bash
cd src
python run.py <config_filename>
``` 
Example for obtaining the BEST RESULTS (*note* that it takes around 15 hours):

```bash
cd src
python run.py experiment_BEST
``` 
A folder with the `config_filename` name will be created under `checkpoints` folder.
Here you will find the weights of the currently trained model. Our best model can be found under `checkpoints/experiment_BEST`

### Prediction
For prediction run the following commands:
```bash
cd src
python predict.py <config_filename> <model> [OPTIONAL_ARGUMENTS]
``` 
Here, `<model>` refers to a a folder name, under `checkpoints` folder,
for the corresponding model weights.

For obtaining the BEST RESULTS you first must download the weights from 
[this](https://drive.google.com/drive/folders/1BWNHppI31-Cc7nggMAW_kVfOhsIhgzFQ?usp=sharing) link and place them under `checkpoints/experiment_BEST`.
and then run the following command:

```bash
cd src
python predict.py experiment_BEST experiment_BEST
``` 

### :chart_with_upwards_trend: Tensorboard
Beside local log files, we use Tensorboard to log our metrics and losses. After each training,
Tensorboard event files will be created under `logs/<config_filename>`. You can run the following command
to check Tensorboard logs on localhost:
```
tensorboard --logdir=logs/<config_filename>
```
## :notebook: Google Colab Notebook
If you cannot get the right CUDA and PyTorch versions locally, or if you have any other compatibility issues,
we also provide a Colab notebook
where the environment is set exactly as on our local machine, for reproducibility. 
However, you must first generate the augmented dataset, locally. You can find
the notebook under `EPFL_ML_project_2.ipynb`. To run it, you must upload the code to your Google Drive account and rename
your PROJECT_ROOT_FOLDER to `EPFL_ML_project_2`.

## :trophy: Results
All submissions will be stored under `results` folder in the file `submission_<config_filename>_avg_patch_ttime_aug.csv`.

The `.csv` uploaded on AICrowd platform can be found under `results/submission_experiment_BEST_avg_patch_ttime_aug.csv`.

These were our best results:

| F1-Score    | Accuracy    |
| ----------- | ----------- |
| 91.7        | 95.5        |

Visual predictions:

![Prediction](doc/overlap_6.png)

## :eyes: References
```
@misc{milesial:2021,
  Author = {milesial},
  Title = {Pytorch-UNet},
  Year = {2021},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/milesial/Pytorch-UNet}}
}
@misc{mbsariyildiz:2019,
  Author = {Mert Bulent Sariyildiz},
  Title = {focal-loss.pytorch},
  Year = {2019},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/mbsariyildiz/focal-loss.pytorch}}
}

```

