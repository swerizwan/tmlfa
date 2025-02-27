# A Transformer-Based Framework for Multi-Task Face Analysis via Multi-Level Fusion and Attention

<img src="https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67" alt="Oryx Video-ChatGPT" data-canonical-src="https://i.imgur.com/waxVImv.png" style="max-width: 100%;">

# Overview

We propose TMLFA, a transformer-based framework for simultaneous facial expression recognition, gender and age estimation, and face attribute analysis. Using a swin transformer, TMLFA extracts facial features, enhanced by a Multi-Level Feature Fusion (MLFF) module and Channel Attention (CA) to highlight key facial details. The Multi-Level Channel Attention (MLCA) ensures efficient task-specific information propagation. Experiments on MS-Celeb-1M, RAF-DB, IMDB+WIKI, and CelebA datasets show that TMLFA outperforms existing methods in facial attribute recognition and estimation tasks.

# 👁️💬 Architecture

The comprehensive pipeline of the TMLFA framework.

<img style="max-width: 100%;" src="https://github.com/swerizwan/tmlfa/blob/main/resources/architecture.jpg" alt="PMRR Overview">

# TMLFA Environment Setup

This repository contains the code for face recognition experiments using PyTorch and other necessary libraries. Follow the instructions below to set up the environment and download the required datasets for running experiments.

The instructions for setting up a Conda environment named `tmlfa` with the required dependencies:

## Requirements

- Python 3.8
```
conda create --no-default-packages -n tmlfa python=3.8
conda activate tmlfa
```

### Install Required Packages

- [PyTorch](https://www.pytorch.org) Tested with version 1.9.0. To install, use the following command:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge opencv
conda install -c anaconda numpy
conda install -c anaconda argparse
pip install timm
pip install tensorboard
```
## Download Datasets

For the experiments in this project, the following datasets are required:

### MS-Celeb-1M Dataset

Download the MS-Celeb-1M dataset from the official source or from [this link](https://exposing.ai/msceleb/).

Once downloaded, extract the dataset to a directory and specify the path in your code.

### RAF-DB Dataset

Download the RAF-DB (Real-world Affective Faces Database) from [this link](http://www.whdeng.cn/RAF/model1.html). Follow the instructions on the website for downloading and extracting the dataset.

### IMDB-WIKI Dataset

Download the IMDB-WIKI dataset from [this GitHub repository](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). This dataset contains face images with age and gender labels. Follow the instructions provided in the repository for dataset extraction.

### CelebA Dataset

Download the CelebA dataset from [this link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). After downloading, extract the dataset and specify the location in your experiment setup.

## Running Experiments

```
python train.py
python demo.py
```

