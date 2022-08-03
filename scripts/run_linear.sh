#!/bin/sh

#SBATCH -o /home/gridsan/ybo/advaug/scripts/outputs/linear_random_chapman.sh.log-%j 
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

python main_linear.py \
    --dataset stl10 \
    --data_dir /datasets/yshi \
    --max_epochs 100 \
    --gpus 0 \
    --lr 0.1 \
    --weight_decay 0 \
    --batch_size 32 \
    --num_workers 10 \
    --name linear_random_chapman \
    --project advaug_test \
    --entity jessica-bo \
    --wandb 