import os

num_devices = 1
output_directory = "/home/advaug_ecg/outputs"
sweep_name = "pretrain/adversarial" 
job_directory = os.path.join(output_directory, sweep_name)

# Make top level directories
if not os.path.exists(job_directory):
    os.mkdir(job_directory)

seeds = [1, 10, 42]
alpha_sparsitys = [0.1, 0.5] 
ratios = [0.8, 1] 
unet_depths = [1]
nmaskss = [1, 2, 3] 
positive_pairings = ["SimCLR", "CMSC"]

batch_size = 32
accumulate_grad_batch = 4
embedding_dim = 512 
lr = 0.0001
adv_lr = 0.0001

wandb_name = "your_account"
project_name = "test_project"

for seed in seeds: 
    for unet_depth in unet_depths:
        for nmasks in nmaskss:
                for alpha_sparsity in alpha_sparsitys:
                    for ratio in ratios:
                        for positive_pairing in positive_pairings:
                            hyperparams_name = "{}_R{}_alpha{}_masks{}_resnet{}_unet{}".format(positive_pairing, ratio, alpha_sparsity, nmasks, embedding_dim, unet_depth)
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


                                fh.writelines("python /home/advaug_ecg/main_pretrain.py \
                                                --method advmask \
                                                --seed {} \
                                                --num_devices {} \
                                                --dataset cinc2020 \
                                                --max_epochs 150 \
                                                --name {}/{} \
                                                --wandb \
                                                --entity {} \
                                                --project {} \
                                                --lr {} \
                                                --adv_lr {} \
                                                --batch_size {} \
                                                --accumulate_grad_batches {} \
                                                --embedding_dim {} \
                                                --alpha_sparsity {} \
                                                --ratio {} \
                                                --unet_depth {} \
                                                --nmasks {} \
                                                --simclr_loss_only \
                                                --positive_pairing {} \n".format(seed, num_devices, sweep_name, hyperparams_name, 
                                                                                 wandb_name, project_name, lr, adv_lr, batch_size, 
                                                                                 accumulate_grad_batch, embedding_dim, alpha_sparsity, 
                                                                                 ratio, unet_depth, nmasks, positive_pairing))

                            os.system("sbatch %s" %job_file)
