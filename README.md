## ESRGAN (TensorFlow)

This repository provides a TensorFlow implementation of the paper "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" by X. Wang et al.

## Dependencies
tensorflow, openCV, sklearn, numpy

The versions of my test environment :  
Python==3.6.8, tensorflow-gpu==1.12.0, openCV==4.1.0,  scikit-learn==0.20.3, numpy==1.16.2

## How to use

#### 1. Prepare data for training

Prepare your data and put them into the directory specified by the flag "data_dir"(e.g. './data/LSUN') of train.py. Other necessary directories are
created automatically as set in the script.

#### 2. Prepare data for training
Run train.py script. The main processes are :
- Data processing : create patches of HR and LR(by downsampling HR pathces). These processed data can be saved in directories so that they can be recycled.

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

## Test Result
Coming soon!


## Limitations

- Only 4x super-resolution is supported
- Grayscale images are not supported
- Only Single GPU usage


## To do list
The following features have not been implemented apart from the paper.

- [x] Perceptual loss using VGG19(currently pixel-wise loss is implemented instead)
- [x] Learning rate scheduling
- [ ] Network interpolation
- [ ] Data augmentation
- [ ] Evaluation metrics

### Notes
Some setting parameters like the number of RRDB blocks, mini-batch size, the number of iteration are changed corresponding to my test environment.
So, please change them if you would prefer the same condition as the paper.


## Reference
* Paper
Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, and Chen Change Loy : ESRGAN: Enhanced Super-ResolutionGenerative Adversarial Networks, ECCV, 2018. http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf


* Official implementation with Pytorch by the paper's authors  
https://github.com/xinntao/BasicSR
