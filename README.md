<div align="center">

# LiDPM: Rethinking Point Diffusion for Lidar Scene Completion
[![Paper](https://img.shields.io/badge/arXiv-2504.17791-brightgreen)](https://arxiv.org/abs/2504.17791)
[![Conference](https://img.shields.io/badge/IEEE_IV-2025-blue)](https://ieee-iv.org/2025/)
[![Project WebPage](https://img.shields.io/badge/Project-webpage-%23fc4d5d)](https://astra-vision.github.io/LiDPM/)

</div>

<p align="center">
  <img src="./media/teaser.png" width="90%" />
</p>

This is an official repository for the paper
```
LiDPM: Rethinking Point Diffusion for Lidar Scene Completion
Tetiana Martyniuk, Gilles Puy, Alexandre Boulch, Renaud Marlet, Raoul de Charette
IEEE IV 2025
```

**Updates:**
- 25/06/2025: training and inference code released.


## Installation

The code uses **Python 3.7**.

#### Create a Conda virtual environment:

```bash
conda create --name lidpm python=3.7
conda activate lidpm
```
#### Clone the project and install requirements:

```bash
git clone https://github.com/astra-vision/LiDPM
cd LiDPM

sudo apt install build-essential libopenblas-dev
pip install -r requirements.txt

# Install Minkowski Engine
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --install-option="--blas=openblas" --install-option="--force_cuda" -v --no-deps
```
## SemanticKITTI Dataset

The SemanticKITTI dataset has to be download from the official [webpage](http://www.semantic-kitti.org/dataset.html#download) and extracted in the following structure:

```
SemanticKITTI
    └── dataset
        └── sequences
            ├── 00/
            │   ├── velodyne/
            |   |       ├── 000000.bin
            |   |       ├── 000001.bin
            |   |       └── ...
            │   └── labels/
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            ├── 08/ # for validation
            ├── 11/ # 11-21 for testing
            └── 21/
                └── ...
```

## Ground truth generation

To generate the complete scenes run the `map_from_scans.py` script. 
This will use the dataset scans and poses to generate the sequence map to be used as the ground truth during training.

Specify the SemanticKITTI path and the output path in the corresponding config file.
```
cd lidpm/scripts
python map_from_scans.py configs/map_from_scans.yaml
```

## Training the model

<p align="center">
  <img src="./media/training_pipeline.png" width="50%" />
</p>

For training the diffusion model, the configurations are defined in `config/train.yaml`, and the training can be started with:

```
cd lidpm
python train.py --config config/train.yaml
```
Don't forget to specify the `data_dir` and `gt_map_dir` in the config file.

The training was performed on 4 NVIDIA A100 GPUs for 40 epochs.

## Inference

<p align="center">
  <img src="./media/inference.png" width="50%" />
</p>

To complete the lidar scans, run
```
cd scripts
python inference.py configs/inference.yaml
```
In the corresponding config you should specify the path to the diffusion checkpoint, dataset path, and output folder, 
and decide if you want to run inference on a particular sequence or over the list of predefined pointclouds 
(configured in `canonical_minival_filename`).

## Citation

If you build upon LiDPM paper or code, please cite the following paper:

```
@INPROCEEDINGS{martyniuk2025lidpm,
      author    = {Martyniuk, Tetiana and Puy, Gilles and Boulch, Alexandre and Marlet, Renaud and de Charette, Raoul},
      booktitle = {2025 IEEE Intelligent Vehicles Symposium (IV)},
      title     = {LiDPM: Rethinking Point Diffusion for Lidar Scene Completion},
      year      = {2025},
    }
```

### Acknowledgments

This code is developed upon the [LiDiff](https://github.com/PRBonn/LiDiff/tree/main) codebase.
We modify it to depart from the "local" diffusion paradigm to the "global" one presented in [LiDPM](https://astra-vision.github.io/LiDPM/) paper.
We thank the authors for making their work publicly available.