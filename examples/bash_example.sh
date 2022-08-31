# Pretrain SimCLR with Gaussian augmentation, saves checkpoints in test_experiment and logs results to wandb
# Add --debug to shorten epoch training 
# Remove --wandb to disable logging 

python /home/advaug_ecg/main_pretrain.py \
    --seed 1 \
    --num_devices 1 \
    --simclr_loss_only \
    --dataset cinc2021 \
    --max_epochs 100 \
    --name test_experiment \
    --wandb \
    --project wandb_project \
    --entity wandb_name \
    --lr 0.01 \
    --batch_size 32 \
    --accumulate_grad_batches 4 \
    --encoder_name resnet \
    --embedding_dim 512 \
    --gaussian \
    --positive_pairing SimCLR \