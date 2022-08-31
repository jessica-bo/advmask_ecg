# advmask_ecg
Adversarial masking for self-supervised pretraining of 12-lead ECGs. 


We adapt parts of our code from the following sources: 
1. Self-Supervised Pre-Training of Networks with CLOCS (https://github.com/danikiyasseh/CLOCS)
2. Adversarial Masking for Self-Supervised Learning (https://github.com/YugeTen/adios)
3. Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram (https://github.com/Jwoo5/fairseq-signals)
4. In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis (https://github.com/seitalab/dnn_ecg_comparison)
5. torch_ecg (https://github.com/DeepPSP/torch_ecg/)


# Datasets 
### CinC2021 (Physionet/Computing in Cardiology 2021)
Download WFDB zipped files (https://moody-challenge.physionet.org/2021/) to `data/cinc2021/raw/` and unzip. Merge WFDB_ChapmanShaoxing and WFDB_Ningbo to a folder named WFDB_ShaoxingUniv.

Generate train/val split by running `python data/cinc2021/save_splits.sh`


### Chapman-Shaoxing
Download dataset (https://figshare.com/collections/ChapmanECG/4560497/2) to `data/chapman/raw/`

Generate train/val/test split by running `python data/chapman/save_splits.sh`

# Run Example
Install requirements `pip install requirements.txt`. You will need a GPU for training with Pytorch-Lightning.

To run a single pretraining trial, run `bash examples/bash_example.sh`

To run a sweep over hyperparameters via slurm, run `python examples/slurm_sweep_example.py`