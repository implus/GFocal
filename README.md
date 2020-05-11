# Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

## Introduction

Advanced one-stage object detectors have gradually focused on the representation of bounding box and its localization quality, where an additional prediction branch is optimized to estimate the quality of bounding box regression under Dirac delta distribution. However, the quality estimation is independently trained (only for positives usually) but compositely utilized with the classification confidence during inference, leading to their weak correlations and possible train-test inconsistencies. Moreover, the widely used Dirac delta distribution has limitations for reflecting the underlying real distribution caused by box ambiguity and uncertainty. Aiming at overcoming these defects, we design to improve the representation of both bounding box and its localization quality: (1) the localization quality and classification score are jointly presented as a single variable, bridging the train-test gap and enabling their strongest correlation;  (2) the arbitrary unimodal box distribution is directly learnt via networks, providing more informative and accurate bounding box regression. To successfully optimize the improved representations with their flexible continuous labels, we propose Generalized Focal Loss (GFL), which generalizes the original Focal Loss from 0/1 discrete formulation to the continuous float version. Specifically, GFL enables learning localization quality (0$\sim$1) simultaneously for the classification vector and provides the single peak supervision around any float target for unimodal box distribution. Without bells and whistles, GFL achieves 45.0\% AP with ResNet101 backbone on COCO {\tt test-dev} set, considerably surpassing previous state-of-the-art Cascade R-CNN (42.8\%) and ATSS (43.6\%), under the same backbone and comparable training settings.


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
--- |:---:|:---:|:---:|:---:|:---:
GFL_R_50_FPN_1x              | No  | 40.2 | 40.3 | [Google]()
GFL_R_101_FPN_2x             | Yes | 44.9 | 45.0 | [Google]()
GFL_dcnv2_R_101_FPN_2x       | Yes | 47.2 | 47.3 | [Google]()
GFL_X_101_32x8d_FPN_2x       | Yes | 45.7 | 46.0 | [Google]()
GFL_dcnv2_X_101_32x8d_FPN_2x | Yes | 48.3 | 48.2 | [Google]()

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
