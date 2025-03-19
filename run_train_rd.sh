#!/bin/bash
#SBATCH --job-name=train-rd
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

# Define parameter lists
alphas=(1 5)
seeds=(8444 174 2654 99965 1025 1222 4785 6444 302 1200)

for alpha in "${alphas[@]}"; do
  for seed in "${seeds[@]}"; do
    wandb_name="Resnet18-Student-alpha${alpha}-seed${seed}"

    python train.py \
      --data-dir /home/gratzerm/datasets/cifar100 \
      --dataset cifar100 \
      --train-split train \
      --val-split test \
      --model resnet18 \
      --img-size 32 \
      --epochs 220 \
      --batch-size 512 \
      --opt sgd \
      --lr 0.08 \
      --sched cosine \
      --weight-decay 0.0005 \
      --momentum 0.9 \
      --warmup-epochs 10 \
      --smoothing 0.1 \
      --drop 0.3 \
      --hflip 0.5 \
      --train-crop-mode rrc \
      --ratio 0.75 1.33 \
      --scale 0.08 1.0 \
      --mixup 0.1 \
      --cutmix 1.0 \
      --color-jitter 0.0 \
      --amp \
      --channels-last \
      --min-lr 0.000001 \
      --crop-pct 0.95 \
      --train-interpolation bicubic \
      --warmup-lr 0.002 \
      --aa rand-m6-mstd0.5-inc1 \
      --checkpoint-hist 2 \
      --num-classes 100 \
      --seed "$seed" \
      --sched-on-updates \
      --log-wandb \
      --experiment CIFAR-100-RD \
      --wandb-name "$wandb_name" \
      --wandb-tags S \
      --wandb-group Student \
      --teacher-model resnet152 \
      --teacher-model-path /home/gratzerm/pytorch-image-models/output/train/20250220-215540CIFAR-100resnet152CIFAR-10032/last.pth.tar \
      --adjusted-training R \
      --alpha "$alpha" \
      --beta 0.9 \
      --temperature 4
  done
done
