## ESRGAN (TensorFlow)

This repository provides a TensorFlow implementation of the paper "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" by X. Wang et al.

## Dependencies
tensorflow, openCV, sklearn, numpy

The versions of my test environment :  
Python==3.6.8, tensorflow-gpu==1.12.0, openCV==4.1.0,  scikit-learn==0.20.3, numpy==1.16.2

## How to Use

#### 1. Prepare data for training

Prepare your data and put them into the directory specified by the flag "data_dir"(e.g. './data/LSUN') of train.py. Other necessary directories are
created automatically as set in the script.

#### 2. Prepare data for training
Run train.py script. The main processes are :
- Data processing : create patches of HR and LR(by downsampling HR patches). These processed data can be saved in directories so that they can be recycled to use.

- Pre-train with pixel-wise loss : As described in the paper, pre-training of Generator is done. You can set "pretrain_generator" flag to False to use an existing pre-trained checkpoint model. (training ESRGAN without pre-trained model is not supported.)

- Training ESRGAN : based on pre-trained model, training ESRGAN is done

```
# python train.py

(data directory can be passed by the optional argument)
# python train.py --data_dir ./data/LSUN
```

#### 3. Inference LR data
After training is finished, super-resolution of LR images is available. Input data can be specified "data_dir" of inference.py script.

```
# python inference.py

(data directory can be passed by the optional argument)
# python inference.py --data_dir ./data/inference
```

#### 4. Inference via Network interpolation
The paper proposes the network interpolation method which linearly combines the weights of pixelwise-based pretrain model and ESRGAN generator. You can run this after training both pre-train model and ESRGAN finishes. Input data can be specified "data_dir" of network_interpolation.py script.

```
# python network_interpolation.py

(data directory can be passed by the optional argument)
# python network_interpolation.py --data_dir ./data/inference
```

## Experiment Result
#### DIV2K dataset
DIV2K is a collection of 2K resolution high quality images. <br>
https://data.vision.ee.ethz.ch/cvl/DIV2K/

<img src="img/0833.png">
<img src="img/0887.png">
<img src="img/0896.png">
from left to right: bicubic interpolation, ESRGAN, ESRGAN with network interpolation, High resolution(GT). 4x super resolution.

#### LSUN
LSUN is a collection of ordinaly resolution bedroom images. <br>
https://www.kaggle.com/jhoward/lsun_bedroom/data

<img src="img/111b822af95747f45f5d25a84f8094c10b27c765.png">
<img src="img/11183b7a2e0ee4be9990721d9ddc7fa34997b41f.png">
<img src="img/1117e6b64a7b4336df58eb351cff435529485e91.png">
from left to right: bicubic interpolation, ESRGAN, ESRGAN with network interpolation, High resolution(GT). 4x super resolution.

#### Experiment condition
- training with 800 images and cropped 2 patches per image for DIV2K
- training with about 5000 images from 20% collection dataset and cropped 2 patches per image for LSUN
- apply data augmentation(horizontal flip and rotate by 90 degree)
- 15 RRDBs, 32 batchsize, 50,000 iteration per training phase. Other parameters are the same as the paper.
- Network interpolation parameter is 0.2


## Limitations

- Only 4x super-resolution is supported
- Grayscale images are not supported
- Only Single GPU usage


## To do list
The following features have not been implemented apart from the paper.

- [x] Perceptual loss using VGG19(currently pixel-wise loss is implemented instead)
- [x] Learning rate scheduling
- [x] Network interpolation
- [x] Data augmentation
- [x] Evaluation metrics

### Notes
Some setting parameters like the number of RRDB blocks, mini-batch size, the number of iteration are changed corresponding to my test environment.
So, please change them if you would prefer the same condition as the paper.


## Reference
* Paper
Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, and Chen Change Loy : ESRGAN: Enhanced Super-ResolutionGenerative Adversarial Networks, ECCV, 2018. http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf


* Official implementation with Pytorch by the paper's authors  
https://github.com/xinntao/BasicSR
