# MSTCN-VAE: Micro-Gesture Classification

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Acknowledgments](#acknowledgments)
8. [Citation](#citation)

---

## Introduction

This repository contains the source code for the MSTCN-VAE model, designed for micro-gesture recognition using skeleton modality. The MSTCN-VAE employs an unsupervised learning approach, offering robust performance on datasets such as iMiGUE.

---

## Prerequisites

To set up and run MSTCN-VAE, ensure you have the following installed:

- Python 3.5 or higher
- [PyTorch](http://pytorch.org/)
- Other dependencies listed in the `requirements.txt` file

Install additional dependencies using:
```bash
pip install -r requirements.txt
```

### Installation

Navigate to the `torchlight` directory and run:
```bash
cd torchlight
python setup.py install
cd ..
```

---

## Data Preparation

Before training and testing, the datasets must be prepared and converted into the proper file structure for efficient loading by the MSTCN-VAE model.

1. **Prepare Initial Data**
   - Place the initial CSV files into:
     ```
     /tools/data/iMiGUE/
     ```
   - Use this directory to generate the necessary `.npy` files for the model.

2. **Generate Original Skeleton (OS) Data**
   - Navigate to the `tools` directory and execute:
     ```bash
     python imigue_gendata.py
     ```

3. **Generate Angle-Extracted (AE) Data**
   - Navigate to the `tools` directory and execute:
     ```bash
     python imigue_polar_gendata.py
     ```

> **Note:** After processing the CSV data, move the generated `.npy` and `.pkl` files to the following directory:
> ```
> /data/iMiGUE/imigue_processed/
> ```

---

## Training

To train a new MSTCN-VAE model on the iMiGUE dataset, run the following command:
```bash
python main.py recognition -c config/mstcn_vae/imigue/train.yaml
```

### Training Outputs

- **Model Weights**
- **Configuration Files**
- **Logs**

These will be saved in the default directory:
```bash
./work_dir
```

### Customizing Training Parameters

You can modify training parameters such as `work_dir`, `batch_size`, `step`, `base_lr`, and `device` either in the configuration file or directly via the command line. The order of priority is:

**Command Line > Config File > Default Parameters**

For detailed usage information, run:
```bash
python main.py -h
```

---

## Evaluation

To evaluate the trained MSTCN-VAE model, run:
```bash
python main.py recognition -c config/mstcn_vae/imigue/test.yaml --weights <path to model weights>
```

---

## Acknowledgments

We thank the developers of MS-G3D, P&C, and TCN for their contributions. We also extend our gratitude to the MiGA Workshop & Challenge 2023 for providing the baseline code that inspired this work.

---

## Citation

If you find MSTCN-VAE helpful in your research, please cite it as follows:

```bibtex
@inproceedings{MSTCNVAE,
  author    = {Wenxuan Yuan and Shanchuan He and Jianwen Dou},
  title     = {MSTCN-VAE: An Unsupervised Learning Method for Micro-Gesture Recognition Based on Skeleton Modality},
  booktitle = {Proceedings of the CEUR Workshop on MiGA Workshop},
  series    = {CEUR Workshop Proceedings},
  volume    = {3522},
  pages     = {Paper 5},
  year      = {2023},
  url       = {https://ceur-ws.org/Vol-3522/paper_5.pdf}
}
```

