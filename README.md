# Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

## Introduction

One-stage detector basically formulates object detection as dense classification and localization (i.e., bounding box regression). The classification is usually optimized by Focal Loss and the box location is commonly learned under Dirac delta distribution. A recent trend for one-stage detectors is to introduce an \emph{individual} prediction branch to estimate the quality of localization, where the predicted quality facilitates the classification to improve detection performance. This paper delves into the \emph{representations} of the above three fundamental elements: quality estimation, classification and localization. Two problems are discovered in existing practices, including (1) the inconsistent usage of the quality estimation and classification between training and inference (i.e., separately trained but compositely used in test) and (2) the inflexible Dirac delta distribution for localization when there is ambiguity and uncertainty which is often the case in complex scenes. To address the problems, we design new representations for these elements. Specifically, we merge the quality estimation into the class prediction vector to form a joint representation of localization quality and classification, and use a vector to represent arbitrary distribution of box locations. The improved representations eliminate the inconsistency risk and accurately depict the flexible distribution in real data, but contain \emph{continuous} labels, which is beyond the scope of Focal Loss. We then propose Generalized Focal Loss (GFL) that generalizes Focal Loss from its discrete form to the \emph{continuous} version for successful optimization. On COCO {\tt test-dev}, GFL achieves 45.0\% AP using ResNet-101 backbone, surpassing state-of-the-art SAPD (43.5\%) and ATSS (43.6\%) with higher or comparable inference speed, under the same backbone and training settings. Notably, our best model can achieve a single-model single-scale AP of 48.2\%, at 10 FPS on a single 2080Ti GPU.


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.


## Get Started

Please see [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of MMDetection.


## Train

```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in 'data/coco/'

./tools/dist_train.sh configs/gfl_r50_1x.py 8 --validate
```

## Inference

```python
./tools/dist_test.sh configs/gfl_r50_1x.py work_dirs/gfl_r50_1x/epoch_12.pth 8 --eval bbox
```

## Models

For your convenience, we provide the following trained models. All models are trained with 16 images in a mini-batch with 8 GPUs.

Model | Multi-scale training | AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:
GFL_R_50_FPN_1x              | No  | 40.2 | 40.3 | [Google](https://drive.google.com/file/d/184HAOoCl6j1-u0ad_lzmFFYQPY9nKxgy/view?usp=sharing)
GFL_R_50_FPN_2x              | Yes | 42.8 | 43.1 | [Google](https://drive.google.com/file/d/1j8doGQDi1w79Ffk4QuxX65y1A3fyraUe/view?usp=sharing)
GFL_R_101_FPN_2x             | Yes | 44.9 | 45.0 | [Google](https://drive.google.com/file/d/1vCYKVejsxO0Fj3CNtjYg_giq3aKmQ8gB/view?usp=sharing)
GFL_dcnv2_R_101_FPN_2x       | Yes | 47.2 | 47.3 | [Google](https://drive.google.com/file/d/1lJT5jj6mU29fLXHFRmMBIqKi-jZGSlSR/view?usp=sharing)
GFL_X_101_32x8d_FPN_2x       | Yes | 45.7 | 46.0 | [Google](https://drive.google.com/file/d/1VqlZKmwVYmmQzU1z-qOHiAlToG8wwIOL/view?usp=sharing)
GFL_dcnv2_X_101_32x8d_FPN_2x | Yes | 48.3 | 48.2 | [Google](https://drive.google.com/file/d/13W38rPTxvxwmQuDR2DIsTxvGrPhIOOOz/view?usp=sharing)

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[2] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[3] *`dcnv2` denotes deformable convolutional networks v2. Note that for ResNet based models, we apply deformable convolutions from stage c3 to c5 in backbones. For ResNeXt based models, only stage c4 and c5 use deformable convolutions.* \
[4] *Refer to more details in config files in `config/`* \



## Acknowledgement

Thanks MMDetection team for the wonderful open source project!


## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{GFocaL,
  title   = {Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection},
  author  = {to appear},
  journal = {to appear},
  year    = {2020}
}
```
