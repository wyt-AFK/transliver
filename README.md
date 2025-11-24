# TransLiver
Code for MICCAI2023 [paper](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_31):  TransLiver: A Hybrid Transformer Model for Multi-phase Liver Lesion Classification.

![miccai model](imgs/miccai_model.png)

TransLiver is a hybrid framework with ViT backbone for liver lesion classification. The repository now targets the [PLC-CECT dataset](https://www.scidb.cn/en/detail?dataSetId=d685a0b9f8974a2a9d7c880be1dc36e9), which provides four CT phases and four liver-related lesion categories:

- Hepatocellular carcinoma (HCC)
- Intrahepatic cholangiocarcinoma (ICC)
- Combined hepatocellular cholangiocarcinoma (cHCC-CCA)
- Non-liver cancer

We design a pre-processing unit to reduce annotation cost by obtaining lesion areas on multi-phase CTs from annotations marked on a single phase. To alleviate the limitations of pure transformers, we propose a multi-stage pyramid structure and add convolutional layers to the original transformer encoder, which helps improve the model performance. Additional cross phase tokens at the last stage complete a multi-phase fusion, focusing on cross-phase communication and improving fusion effectiveness as compared with conventional modes.

## Requirements

We use Python 3.9.12 in our project. The main packages are included in `requirements.txt`.

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
GPU training is recommended. Install the CUDA-enabled versions of PyTorch and related packages that match your system if the defaults in `requirements.txt` do not.

## Dataset

The project targets the [PLC-CECT dataset](https://www.scidb.cn/en/detail?dataSetId=d685a0b9f8974a2a9d7c880be1dc36e9), which provides four phases per study and four lesion types (HCC, ICC, cHCC-CCA, Non-liver cancer).

1. Download the dataset from the link above and extract it locally.
2. Convert the CT volumes and lesion masks for each phase to `.nii.gz` if they are not already provided in that format.
3. Organize each phase (e.g., `arterial`, `portal`, `delay`, `non-contrast`) into separate folders such as:

   ```
   /dataset/arterial
   /dataset/portal
   /dataset/delay
   /dataset/noncontrast
   ```

4. Place the corresponding lesion masks in phase-specific directories that mirror the image layout, e.g., `/dataset/arterial_label`, `/dataset/portal_label`, etc.
5. Create a class label JSON file mapping each lesion ID to its class ID (0–3) and store it in `/path/to/lesions`, which is the directory of pre-processed lesion crops.

The scripts in `./register/reg_preprocess.py`, `./register/reg_postprocess.py`, and `./classification/preprocess.py` illustrate expected data formats and can be adapted if your layout differs.

## Getting Started

The data paths and hyperparameters should be changed according to your own project. Please see `./register/config.py` and `./classification/config.py`.

### Pre-processing

#### register

The voxelmorph code in pytorch is in reference of [VoxelMorph-torch](https://github.com/zuzhiang/VoxelMorph-torch).

- `./register/reg_preprocess.py`: preprocess for register
- `./register/reg_train.py`: register train
- `./register/reg_test.py`: register all data
- `./register/reg_postprocess.py`: lesion matcher

#### classification

`./classification/preprocess.py`: preprocess for classification. Update the paths in `classification/config.py` to point to your processed lesion crops and label JSON before running:

```bash
python classification/preprocess.py
```

### Train

Get pretrain weights of [CMT-S](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/cmt_pytorch) in `./pre`.

Adjust hyperparameters and dataset locations in `classification/config.py`, then start training:

```bash
bash classification/run.sh
```

The default configuration trains a four-class model consistent with PLC-CECT labels.

### Inference

Update checkpoint and data paths in `classification/config.py`, then run:

```bash
python classification/test.py
```

## BibTeX

```
@inproceedings{wang2023transliver,
  title={TransLiver: A Hybrid Transformer Model for Multi-phase Liver Lesion Classification},
  author={Wang, Xierui and Ying, Hanning and Xu, Xiaoyin and Cai, Xiujun and Zhang, Min},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={329--338},
  year={2023},
  organization={Springer}
}
```
