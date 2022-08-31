"""
https://vsoch.github.io/lessons/sherlock-jobs/
"""
#!/usr/bin/env python

import os

num_devices = 1
output_directory = "/home/gridsan/ybo/advaug/outputs/"
sweep_name = "pretrain/adversarial_rlm" # "pretrain_chapman_resnet_augmentations" # 

job_directory = os.path.join(output_directory, sweep_name)

# Make top level directories
if not os.path.exists(job_directory):
    os.mkdir(job_directory)

seeds = [1, 10, 42]
simclr_loss_flags = ["--simclr_loss_only"]
lrs = [0.0001] 
mask_lrs = [0.0001]
train_mask_intervals = [1] 
alpha_sparsitys = [0.1] #, 0.5] 
ratios = [1] #R1 for masking, R0.5 for fourier
unet_depths = [1]
unet_sizes = ["--unet_large"]
nmaskss = [2] #2 for masking
positive_pairings = ["SimCLR", "CMSC"]
dropouts = [0, 0.2]

fourier_flag = ""#"--fourier"
augmentation_flag = "--rlm"

batch_size = 32
accumulate_grad_batch = 4
embedding_dim = 512 
for seed in seeds: 
    for lr in lrs:
        for mask_lr in mask_lrs: 
            for unet_depth in unet_depths:
                for unet_size in unet_sizes:
                    for nmasks in nmaskss:
                        for simclr_loss_flag in simclr_loss_flags:
                            for alpha_sparsity in alpha_sparsitys:
                                for ratio in ratios:
                                    for positive_pairing in positive_pairings:
                                        for dropout in dropouts:

                                            hyperparams_name = "{}_R{}_drop{}_alpha{}_masks{}_resnet{}_unet{}{}{}{}".format(positive_pairing, ratio, dropout, alpha_sparsity, nmasks, embedding_dim, unet_depth, unet_size, simclr_loss_flag, fourier_flag)
                                            job_path = os.path.join(job_directory, hyperparams_name)
                                            job_file = os.path.join(job_path, "sweep_job.sh")

                                            # Create lizard directories
                                            if not os.path.exists(job_path):
                                                os.mkdir(job_path)

                                            job_seed_path = os.path.join(job_path, "seed{}".format(seed))
                                            
                                            if not os.path.exists(job_seed_path):
                                                os.mkdir(job_seed_path)
                                                print(job_seed_path)
                                                job_file = os.path.join(job_seed_path, "sweep_job.sh")
                                            else:
                                                continue

                                            with open(job_file, 'w') as fh:
                                                fh.writelines("#!/bin/sh\n")
                                                fh.writelines("#SBATCH -o {}/run_log.sh.log-%j \n".format(job_seed_path))
                                                fh.writelines("#SBATCH -c 20 \n")
                                                fh.writelines("#SBATCH --gres=gpu:volta:{} \n".format(num_devices))
                                                fh.writelines("#SBATCH --mail-type=END \n")
                                                fh.writelines("#SBATCH --mail-user=yibo@bwh.harvard.edu \n")

                                                fh.writelines("python /home/gridsan/ybo/advaug/main_pretrain.py \
                                                                --method adversarial \
                                                                --seed {} \
                                                                --num_devices {} \
                                                                --dataset cinc2021 \
                                                                --max_epochs 150 \
                                                                --name {}/{} \
                                                                --project advaug_test \
                                                                --entity jessica-bo \
                                                                --wandb \
                                                                --lr {} \
                                                                --mask_lr {} \
                                                                --batch_size {} \
                                                                --accumulate_grad_batches {} \
                                                                --encoder_name resnet \
                                                                --embedding_dim {} \
                                                                --train_mask_interval 1 \
                                                                --alpha_sparsity {} \
                                                                --ratio {} \
                                                                --unet_depth {} \
                                                                --nmasks {} \
                                                                {} \
                                                                {} \
                                                                {} \
                                                                {} \
                                                                --positive_pairing {} \
                                                                --dropout {} \n".format(seed, num_devices, sweep_name, hyperparams_name, lr, mask_lr, 
                                                                                        batch_size, accumulate_grad_batch, embedding_dim, alpha_sparsity, 
                                                                                        ratio, unet_depth, nmasks, unet_size, simclr_loss_flag, fourier_flag,
                                                                                        augmentation_flag, positive_pairing, dropout))

                                            os.system("sbatch %s" %job_file)
