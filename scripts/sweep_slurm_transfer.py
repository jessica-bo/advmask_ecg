"""
https://vsoch.github.io/lessons/sherlock-jobs/
"""
#!/usr/bin/env python

import os

output_directory = "/home/gridsan/ybo/advaug/outputs/"
dataset = "chapman"
pretrain_folder = "random"#"adversarial_3kg"
pretrain_path = None#os.path.join("/home/gridsan/ybo/advaug/outputs/pretrain/", pretrain_folder)
pretrained_feature_extractors = [None]#[item for item in os.listdir(pretrain_path) if os.path.isdir(os.path.join(pretrain_path, item)) if "--simclr" in item] #[""] #

seeds = [1, 10, 42]
lrs = [0.001] 
batch_sizes = [256] 
backbones = ['resnet']
embedding_dims = [1024] 

reduce_datasets = ["", "0.1", "0.01"]

for reduce_dataset in reduce_datasets:
    trials = ["{}linear".format(reduce_dataset), "{}finetune".format(reduce_dataset)] #, "finetune"]

    for trial in trials:
        sweep_name = "transfer/{}/{}".format(trial, pretrain_folder)
        finetune_flag = "--finetune" if "finetune" in trial else ""

        job_directory = os.path.join(output_directory, sweep_name)
        if not os.path.exists(job_directory):
            os.mkdir(job_directory)

        for seed in seeds: 
            for lr in lrs:
                for batch_size in batch_sizes:
                    for backbone in backbones: 
                        for embedding_dim in embedding_dims:
                            for pretrained_feature_extractor in pretrained_feature_extractors:

                                hyperparams_name = ""#"{}".format(pretrained_feature_extractor)                
                                pretrained_feature_extractor = None#os.path.join(pretrain_path, pretrained_feature_extractor, "seed{}".format(seed))

                                reduce_dataset_flag = reduce_dataset if reduce_dataset=="" else "--reduce_dataset {}".format(reduce_dataset)

                                job_path = os.path.join(job_directory, hyperparams_name)
                                job_file = os.path.join(job_path, "sweep_job.sh")

                                # Create directories
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
                                    fh.writelines("#SBATCH --gres=gpu:volta:1 \n")

                                    # --finetune \
                                    fh.writelines("python /home/gridsan/ybo/advaug/main_transfer.py \
                                                --seed {} \
                                                {} \
                                                --dataset {} \
                                                --max_epochs 200 \
                                                --num_workers 10 \
                                                --name {}/{} \
                                                --project advaug_test \
                                                --entity jessica-bo \
                                                --wandb \
                                                --lr {}\
                                                --batch_size {} \
                                                --pretrained_feature_extractor {} \
                                                --embedding_dim {} \
                                                {} \
                                                \n".format(seed, finetune_flag,  dataset, sweep_name, hyperparams_name, lr, 
                                                           batch_size, pretrained_feature_extractor, embedding_dim, reduce_dataset_flag))

                                os.system("sbatch %s" %job_file)