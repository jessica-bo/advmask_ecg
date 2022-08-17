#!/bin/sh

#SBATCH -o /home/gridsan/ybo/advaug/outputs/linear_random_chapman.sh.log-%j 
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

module load anaconda/2022a


python main_linear.py \
    --dataset chapman \
    --max_epochs 150 \
    --lr 0.001 \
    --weight_decay 0 \
    --batch_size 32 \
    --num_workers 10 \
    --name random_chapman \
    --project advaug_test \
    --entity jessica-bo \
    --wandb 