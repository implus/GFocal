# Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

See more comments in [大白话 Generalized Focal Loss(知乎)](https://zhuanlan.zhihu.com/p/147691786)

[2020.7] GFocal is officially included in [MMDetection V2](https://github.com/open-mmlab/mmdetection/blob/master/configs/gfl/README.md), many thanks to [@ZwwWayne](https://github.com/ZwwWayne) and [@hellock](https://github.com/hellock) for helping migrating the code.

[2020.9] The winner (1st) of GigaVision (object detection and tracking) in ECCV 2020 workshop from DeepBlueAI team adopt GFocal in their [solutions](http://tech.sina.com.cn/csj/2020-09-01/doc-iivhvpwy4252172.shtml).

## Introduction

One-stage detector basically formulates object detection as dense classification and localization (i.e., bounding box regression). The classification is usually optimized by Focal Loss and the box location is commonly learned under Dirac delta distribution. A recent trend for one-stage detectors is to introduce an \emph{individual} prediction branch to estimate the quality of localization, where the predicted quality facilitates the classification to improve detection performance. This paper delves into the \emph{representations} of the above three fundamental elements: quality estimation, classification and localization. Two problems are discovered in existing practices, including (1) the inconsistent usage of the quality estimation and classification between training and inference (i.e., separately trained but compositely used in test) and (2) the inflexible Dirac delta distribution for localization when there is ambiguity and uncertainty which is often the case in complex scenes. To address the problems, we design new representations for these elements. Specifically, we merge the quality estimation into the class prediction vector to form a joint representation of localization quality and classification, and use a vector to represent arbitrary distribution of box locations. The improved representations eliminate the inconsistency risk and accurately depict the flexible distribution in real data, but contain \emph{continuous} labels, which is beyond the scope of Focal Loss. We then propose Generalized Focal Loss (GFL) that generalizes Focal Loss from its discrete form to the \emph{continuous} version for successful optimization. On COCO {\tt test-dev}, GFL achieves 45.0\% AP using ResNet-101 backbone, surpassing state-of-the-art SAPD (43.5\%) and ATSS (43.6\%) with higher or comparable inference speed, under the same backbone and training settings. Notably, our best model can achieve a single-model single-scale AP of 48.2\%, at 10 FPS on a single 2080Ti GPU.

<img src="https://github.com/implus/GFocal/blob/master/gfocal.png" width="1000" height="300" align="middle"/>

For details see [GFocal](https://arxiv.org/pdf/2006.04388.pdf). The speed-accuracy trade-off is as follows:

<img src="https://github.com/implus/GFocal/blob/master/sota_time_acc.jpg" width="541" height="365" align="middle"/>


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

## Speed Test (FPS)

```python
CUDA_VISIBLE_DEVICES=0 python3 ./tools/benchmark.py configs/gfl_r50_1x.py work_dirs/gfl_r50_1x/epoch_12.pth
```

## Models

For your convenience, we provide the following trained models. All models are trained with 16 images in a mini-batch with 8 GPUs.

Model | Multi-scale training | AP (minival) | AP (test-dev) | FPS | Link
--- |:---:|:---:|:---:|:---:|:---:
GFL_R_50_FPN_1x              | No  | 40.2 | 40.3 | 19.4 | [Google](https://drive.google.com/file/d/184HAOoCl6j1-u0ad_lzmFFYQPY9nKxgy/view?usp=sharing)
GFL_R_50_FPN_2x              | Yes | 42.8 | 43.1 | 19.4 | [Google](https://drive.google.com/file/d/1j8doGQDi1w79Ffk4QuxX65y1A3fyraUe/view?usp=sharing)
GFL_R_101_FPN_2x             | Yes | 44.9 | 45.0 | 14.6 | [Google](https://drive.google.com/file/d/1vCYKVejsxO0Fj3CNtjYg_giq3aKmQ8gB/view?usp=sharing)
GFL_dcnv2_R_101_FPN_2x       | Yes | 47.2 | 47.3 | 12.7 | [Google](https://drive.google.com/file/d/1lJT5jj6mU29fLXHFRmMBIqKi-jZGSlSR/view?usp=sharing)
GFL_X_101_32x4d_FPN_2x       | Yes | 45.7 | 46.0 | 12.2 | [Google](https://drive.google.com/file/d/1VqlZKmwVYmmQzU1z-qOHiAlToG8wwIOL/view?usp=sharing)
GFL_dcnv2_X_101_32x4d_FPN_2x | Yes | 48.3 | 48.2 | 10.0 | [Google](https://drive.google.com/file/d/13W38rPTxvxwmQuDR2DIsTxvGrPhIOOOz/view?usp=sharing)

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[2] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[3] *`dcnv2` denotes deformable convolutional networks v2. Note that for ResNe(X)t based models, we apply deformable convolutions from stage c3 to c5 in backbones.* \
[4] *Refer to more details in config files in `config/`.* \
[5] *FPS is tested with a single GeForce RTX 2080Ti GPU, using a batch size of 1.* 


## Acknowledgement

Thanks MMDetection team for the wonderful open source project!


## Citation

If you find GFL useful in your research, please consider citing this project.

```
@article{li2020generalized,
  title={Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2006.04388},
  year={2020}
}
```
