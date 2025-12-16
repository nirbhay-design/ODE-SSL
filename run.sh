#!/bin/bash 

# nohup python train.py --config configs/nodel.c10.yaml --gpu 1 --model resnet50 --epochs 600 --epochs_lin 100 --save_path nodel.c10.r50.pth > logs/nodel.c10.r50.log &

# nohup python train.py --config configs/nodel.c10.yaml --gpu 1 --model resnet18 --epochs 600 --epochs_lin 100 --save_path nodel.c10.r18.pth > logs/nodel.c10.r18.log &

# nohup python train.py --config configs/carl.c10.yaml --gpu 1 --model resnet50 --epochs 1000 --epochs_lin 100 --save_path carl.c10.r50.pth > logs/carl.c10.r50.log &

# nohup python train.py --config configs/carl.c10.yaml --gpu 0 --model resnet18 --epochs 1000 --epochs_lin 100 --save_path carl.c10.r18.pth > logs/carl.c10.r18.log &

# nohup python train.py --config configs/nodel.c100.yaml --gpu 7 --model resnet50 --epochs 600 --epochs_lin 100 --save_path nodel.c100.r50.pth > logs/nodel.c100.r50.log &

# nohup python train.py --config configs/nodel.c100.yaml --gpu 1 --model resnet18 --epochs 600 --epochs_lin 100 --save_path nodel.c100.r18.pth > logs/nodel.c100.r18.log &

# nohup python train.py --config configs/carl.c100.yaml --gpu 2 --model resnet50 --epochs 1000 --epochs_lin 100 --save_path carl.c100.r50.pth > logs/carl.c100.r50.log &

# nohup python train.py --config configs/carl.c100.yaml --gpu 1 --model resnet18 --epochs 1000 --epochs_lin 100 --save_path carl.c100.r18.pth > logs/carl.c100.r18.log &

# nohup python train.py --config configs/nodel.c10.yaml --gpu 2 --model resnet50 --epochs 600 --epochs_lin 100 --save_path nodel.c10.r50.ode5.pth --ode_steps 5 > logs/nodel.c10.r50.ode5.log &

# nohup python train.py --config configs/nodel.c10.yaml --gpu 3 --model resnet18 --epochs 600 --epochs_lin 100 --save_path nodel.c10.r18.ode5.pth --ode_steps 5 > logs/nodel.c10.r18.ode5.log &

# running experiments for less ode steps 

# nohup python train.py --config configs/nodel.c10.yaml --gpu 2 --model resnet50 --epochs 600 --epochs_lin 100 --save_path nodel.c10.r50.ode5.pth --ode_steps 5 > logs/nodel.c10.r50.ode5.log &

# nohup python train.py --config configs/nodel.c10.yaml --gpu 3 --model resnet18 --epochs 600 --epochs_lin 100 --save_path nodel.c10.r18.ode5.pth --ode_steps 5 > logs/nodel.c10.r18.ode5.log &

# nohup python train.py --config configs/nodel.c100.yaml --gpu 0 --model resnet50 --epochs 600 --epochs_lin 100 --save_path nodel.c100.r50.ode5.pth --ode_steps 5 > logs/nodel.c100.r50.ode5.log &

# nohup python train.py --config configs/nodel.c100.yaml --gpu 1 --model resnet18 --epochs 600 --epochs_lin 100 --save_path nodel.c100.r18.ode5.pth --ode_steps 5 > logs/nodel.c100.r18.ode5.log &

# nohup python train.py --config configs/carl.c10.yaml --gpu 4 --model resnet50 --epochs 1000 --epochs_lin 100 --save_path carl.c10.r50.ode5.pth --ode_steps 5 > logs/carl.c10.r50.ode5.log &

# nohup python train.py --config configs/carl.c10.yaml --gpu 1 --model resnet18 --epochs 1000 --epochs_lin 100 --save_path carl.c10.r18.ode5.pth --ode_steps 5 > logs/carl.c10.r18.ode5.log &

# nohup python train.py --config configs/carl.c100.yaml --gpu 5 --model resnet50 --epochs 1000 --epochs_lin 100 --save_path carl.c100.r50.ode5.pth --ode_steps 5 > logs/carl.c100.r50.ode5.log &

# nohup python train.py --config configs/carl.c100.yaml --gpu 6 --model resnet18 --epochs 1000 --epochs_lin 100 --save_path carl.c100.r18.ode5.pth --ode_steps 5 > logs/carl.c100.r18.ode5.log &

###############################################

# nohup python train.py --config configs/florel.c10.yaml --gpu 6 --model resnet18 --epochs 600 --epochs_lin 100 --save_path florel.c10.r18.pth > logs/florel.c10.r18.log &


# ran these experiments after 
# - using liner layer for projection (for NODEL)
# - removing random resized crop augmentation
# - using spectral norm in one neural ODE

# nohup python train.py --config configs/nodel.c10.yaml --gpu 0 --model resnet18 --epochs 350 --epochs_lin 100 --save_path nodel.c10.r18.e350.pth > logs/nodel.c10.r18.e350.log &

# nohup python train.py --config configs/carl.c10.yaml --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --save_path carl.c10.r18.e800.pth > logs/carl.c10.r18.e800.log &

# nohup python train.py --config configs/nodel.c10.yaml --gpu 0 --model resnet18 --epochs 350 --epochs_lin 100 --save_path nodel.c10.r18.e350.adamw.lr0.001.pth --opt AdamW --lr 0.001 > logs/nodel.c10.r18.e350.adamw.lr0.001.log &

# nohup python train.py --config configs/carl.c10.yaml --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --save_path carl.c10.r18.e800.adamw.lr0.001.pth --opt AdamW --lr 0.001 > logs/carl.c10.r18.e800.adamw.lr0.001.log  &

# nohup python train.py --config configs/nodel.c10.yaml --gpu 1 --model resnet18 --epochs 500 --epochs_lin 100 --save_path nodel.c10.r18.e500.adamw.lr0.001.pth --opt AdamW --lr 0.001 > logs/nodel.c10.r18.e500.adamw.lr0.001.log &

# nohup python train.py --config configs/nodel.c10.yaml --gpu 1 --model resnet18 --epochs 350 --epochs_lin 100 --save_path nodel.c10.r18.e350.adamw.lr0.01.pth --opt AdamW --lr 0.01 > logs/nodel.c10.r18.e350.adamw.lr0.01.log &

# nohup python train.py --config configs/carl.c10.yaml --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --save_path carl.c10.r18.e800.adamw.lr0.01.pth --opt AdamW --lr 0.01 > logs/carl.c10.r18.e800.adamw.lr0.01.log  &


nohup python train.py --config configs/lema.c10.yaml --gpu 0 --model resnet50 --epochs 350 --epochs_lin 100 --save_path lema.c10.r50.v2.pth > logs/lema.c10.r50.v2.log &

nohup python train.py --config configs/lema.c10.yaml --gpu 6 --model resnet18 --epochs 350 --epochs_lin 100 --save_path lema.c10.r18.v2.pth > logs/lema.c10.r18.v2.log &

nohup python train.py --config configs/lema.c100.yaml --gpu 1 --model resnet50 --epochs 350 --epochs_lin 100 --save_path lema.c100.r50.v2.pth > logs/lema.c100.r50.v2.log &

nohup python train.py --config configs/lema.c100.yaml --gpu 6 --model resnet18 --epochs 350 --epochs_lin 100 --save_path lema.c100.r18.v2.pth > logs/lema.c100.r18.v2.log &