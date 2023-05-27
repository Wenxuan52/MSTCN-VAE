# ST-GCN baseline for iMiGUE classification track

## Introduction
This repository holds implementation of the paper STGCN:

Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018.

on the iMiGUE dataset:

iMiGUE: An Identity-Free Video Dataset for Micro-Gesture Understanding and Emotion Analysis. Xin Liu, Henglin Shi, Haoyu Chen, Zitong Yu, Xiaobai Li, Guoying Zhao; CVPR, 2021

## Prerequisites
- Python3 (>3.5)
- [PyTorch](http://pytorch.org/)
- Other Python libraries can be installed by `pip install -r requirements.txt`

### Installation
``` shell
cd torchlight; python setup.py install; cd ..
```

## Data Preparation

We show how to prepare the micro-gesture datasts: **iMiGUE**.
Before training and testing, for convenience of fast data loading,
the datasets should be converted to proper file structure that can fit all the STGCN-based models.

Download the training and validating datasets of iMiGUE from codalab, named 'datasets for phase 1 for the classification track'.

You can find the pre-processed data we prepared from the below path:

st-gcn_imigue/data/iMiGUE/imigue_processed

Otherwise, for processing raw data from iMiGUE datasets by yourself,
please refer to below guidances.

#### iMiGUE dataset processing

After uncompressing 'datasets for phase 1 for the online recognition track' to 'data' folder, rebuild the database by this command:

python tools/imigue_gendata.py

## Training
To train a new ST-GCN model, run

```
python main.py recognition -c config/st_gcn/imigue/train.yaml
```
The training results, including **model weights**, configurations and logging files, will be saved under the ```./work_dir``` by default.

You can modify the training parameters such as ```work_dir```, ```batch_size```, ```step```, ```base_lr``` and ```device``` in the command line or configuration files. The order of priority is:  command line > config file > default parameter. For more information, use ```main.py -h```.

Finally, custom model evaluation can be achieved by this command as we mentioned above:
```
python main.py recognition -c config/st_gcn/imigue/test.yaml --weights <path to model weights>
```


<!-- Similary, the configuration file for testing baseline models can be found under the ```./config/baseline```. -->

To speed up evaluation by multi-gpu inference or modify batch size for reducing the memory cost, set ```--test_batch_size``` and ```--device``` like:
```
python main.py recognition -c <config file> --test_batch_size <batch size> --device <gpu0> <gpu1> ...
```


## Citation
Please cite the following paper if you use this repository in your reseach.
```
iMiGUE: An Identity-Free Video Dataset for Micro-Gesture Understanding and Emotion Analysis. Xin Liu, Henglin Shi, Haoyu Chen, Zitong Yu, Xiaobai Li, Guoying Zhao; CVPR, 2021
```


## Contact
For any question regarding the MiGA-IJCAI2023 challenge, feel free to contact
```
Haoyu Chen: chen.haoyu@oulu.fi
```
