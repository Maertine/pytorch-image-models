#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=14-00:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1 # timm does not allow more than 1
#SBATCH --gres=gpu:nvidia_rtx_a4000:1 # Specific GPU
#SBATCH --mem=16gb # memory
#SBATCH --ntasks-per-node=1 # larger than 1 can be used only for small batches
#SBATCH --cpus-per-task=4 # resources for data load, not bottleneck
#SBATCH --output=/home/gratzerm/output/cache_%j.out
# -------------------------------

source .venv/bin/activate

python train.py --data-dir /home/gratzerm/datasets/cifar100 --dataset cifar100 --train-split train --val-split test --model resnet18 --img-size 32 --epochs 220 --batch-size 512 --opt sgd --lr 0.08 --sched cosine --weight-decay 0.0005 --momentum 0.9 --warmup-epochs 10 --smoothing 0.1 --drop 0.3 --hflip 0.5 --train-crop-mode rrc --ratio 0.75 1.33 --scale 0.08 1.0 --mixup 0.1 --cutmix 1.0 --color-jitter 0.0 --amp --channels-last --min-lr 0.000001 --crop-pct 0.95 --train-interpolation bicubic --warmup-lr 0.002 --aa rand-m6-mstd0.5-inc1 --checkpoint-hist 2 --num-classes 100 --seed 567 --sched-on-updates --log-wandb --experiment CIFAR-100 --wandb-name Resnet18-Vanilla --wandb-tags V --wandb-group Vanilla
