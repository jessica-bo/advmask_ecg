"""
https://vsoch.github.io/lessons/sherlock-jobs/
"""
#!/usr/bin/env python

import os

num_devices = 1
output_directory = "/home/gridsan/ybo/advaug/outputs/"
sweep_name = "pretrain/adversarial_cinc2021_resnet" # "pretrain_chapman_resnet_augmentations" # 
job_directory = os.path.join(output_directory, sweep_name)

# Make top level directories
if not os.path.exists(job_directory):
    os.mkdir(job_directory)

seeds = [0, 1, 10, 42]
lrs = [0.0001] 
mask_lrs = [0.0001]
batch_sizes = [32] 
accumulate_grad_batches = [4]
embedding_dims = [512] #1024
train_mask_intervals = [1] 
alpha_sparsitys = [0.1, 0.5] 
ratios = [5, 2, 1]

for seed in seeds: 
    for lr in lrs:
        for batch_size in batch_sizes:
            for accumulate_grad_batch in accumulate_grad_batches:
                for mask_lr in mask_lrs: 
                    for embedding_dim in embedding_dims:
                        for train_mask_interval in train_mask_intervals:
                            for alpha_sparsity in alpha_sparsitys:
                                for ratio in ratios:

                                    hyperparams_name = "lr{}_mlr{}_R{}_alpha{}_int{}_bs{}x{}_resnet{}".format(lr, mask_lr, ratio, alpha_sparsity, train_mask_interval, batch_size, accumulate_grad_batch,embedding_dim)

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
                                                        --train_mask_interval {} \
                                                        --alpha_sparsity {} \
                                                        --ratio {} \n".format(seed, num_devices, sweep_name, hyperparams_name, lr, mask_lr, 
                                                                            batch_size, accumulate_grad_batch, embedding_dim, 
                                                                            train_mask_interval, alpha_sparsity, ratio))

                                    os.system("sbatch %s" %job_file)

