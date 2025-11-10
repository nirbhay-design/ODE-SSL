#!/bin/bash 

# nohup python train.py --config configs/nodel.c10.yaml --gpu 0 --model resnet50 --epochs 600 --epochs_lin 100 --save_path nodel.c10.r50.pth > logs/nodel.c10.r50.log &

# nohup python train.py --config configs/nodel.c10.yaml --gpu 0 --model resnet18 --epochs 600 --epochs_lin 100 --save_path nodel.c10.r18.pth > logs/nodel.c10.r18.log &

# nohup python train.py --config configs/carl.c10.yaml --gpu 1 --model resnet50 --epochs 1000 --epochs_lin 100 --save_path carl.c10.r50.pth > logs/carl.c10.r50.log &

# nohup python train.py --config configs/carl.c10.yaml --gpu 0 --model resnet18 --epochs 1000 --epochs_lin 100 --save_path carl.c10.r18.pth > logs/carl.c10.r18.log &


# ran these experiments after 
# - using liner layer for projection
# - removing random resized crop augmentation

nohup python train.py --config configs/nodel.c10.yaml --gpu 0 --model resnet18 --epochs 350 --epochs_lin 100 --save_path nodel.c10.r18.e350.pth > logs/nodel.c10.r18.e350.log &

nohup python train.py --config configs/carl.c10.yaml --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --save_path carl.c10.r18.e800.pth > logs/carl.c10.r18.e800.log &

nohup python train.py --config configs/nodel.c10.yaml --gpu 0 --model resnet18 --epochs 350 --epochs_lin 100 --save_path nodel.c10.r18.e350.adamw.lr0.001.pth --opt AdamW --lr 0.001 > logs/nodel.c10.r18.e350.adamw.lr0.001.log &

nohup python train.py --config configs/carl.c10.yaml --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --save_path carl.c10.r18.e800.adamw.lr0.001.pth --opt AdamW --lr 0.001 > logs/carl.c10.r18.e800.adamw.lr0.001.log  &