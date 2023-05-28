# MSTCN-VAE model for micro-gesture classification

## Introduction

This repository holds source of the MSTCN-VAE model.

## Prerequisites

- Python3 (>3.5)
- [PyTorch](http://pytorch.org/)
- Other Python libraries can be installed by `pip install -r requirements.txt`

### Installation

``` shell
cd torchlight; python setup.py install; cd ..
```

## Data Preparation

Before training and testing, for convenience of fast data loading, 

the datasets should be converted to proper file structure that can fit all the MSTCN-VAE models.

Firstly, you need put the initial csv file into:

`/tools/data/iMiGUE/`

to generate data for model which is .npy file.

And then, for generating original skeleton (OS) data, 

you need go back to the `tools` directory and run **imigue_gendata.py**.

For generating angle information extracted (AE) data, 

you need go back to the `tools` directory and run **imigue_polar_gendata.py**.

*<Note: After processing the csv data, you need put .npy and .pkl flies into `/data/iMiGUE/imigue_processed/`>*

## Training

To train a new MSTCN-VAE model on iMiGUE dataset, run

```
python main.py recognition -c config/mstcn_vae/imigue/train.yaml
```

The training results, including **model weights**, configurations and logging files, will be saved under the ```./work_dir``` by default.

You can modify the training parameters such as ```work_dir```, ```batch_size```, ```step```, ```base_lr``` and ```device``` in the command line or configuration files. The order of priority is:  command line > config file > default parameter. For more information, use ```main.py -h```.

Finally, custom model evaluation can be achieved by this command as we mentioned above:

```
python main.py recognition -c config/mstcn_vae/imigue/test.yaml --weights <path to model weights>
```

## Acknowledgments

Thanks to the developers of MS_G3D, P&C and TCN. And thanks to MiGA Workshop & Challenge 2023 for providing the baseline code.
