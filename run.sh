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


# nohup python train.py --config configs/lema.c10.yaml --gpu 0 --model resnet50 --epochs 350 --epochs_lin 100 --save_path lema.c10.r50.v2.pth > logs/lema.c10.r50.v2.log &

# nohup python train.py --config configs/lema.c10.yaml --gpu 6 --model resnet18 --epochs 350 --epochs_lin 100 --save_path lema.c10.r18.v2.pth > logs/lema.c10.r18.v2.log &

# nohup python train.py --config configs/lema.c100.yaml --gpu 1 --model resnet50 --epochs 350 --epochs_lin 100 --save_path lema.c100.r50.v2.pth > logs/lema.c100.r50.v2.log &

# nohup python train.py --config configs/lema.c100.yaml --gpu 6 --model resnet18 --epochs 350 --epochs_lin 100 --save_path lema.c100.r18.v2.pth > logs/lema.c100.r18.v2.log &


# nohup python train.py --config configs/lema.c10.yaml --gpu 0 --model resnet50 --epochs 500 --epochs_lin 100 --save_path lema.c10.r50.e500.pth > logs/lema.c10.r50.e500.log &

# nohup python train.py --config configs/lema.c10.yaml --gpu 1 --model resnet18 --epochs 500 --epochs_lin 100 --save_path lema.c10.r18.e500.pth > logs/lema.c10.r18.e500.log &

# nohup python train.py --config configs/lema.c100.yaml --gpu 0 --model resnet50 --epochs 500 --epochs_lin 100 --save_path lema.c100.r50.e500.pth > logs/lema.c100.r50.e500.log &

# nohup python train.py --config configs/lema.c100.yaml --gpu 1 --model resnet18 --epochs 500 --epochs_lin 100 --save_path lema.c100.r18.e500.pth > logs/lema.c100.r18.e500.log &

# nohup python train.py --config configs/lema.c10.yaml --gpu 0 --model resnet50 --epochs 350 --epochs_lin 100 --save_path lema.c10.r50.v3.pth > logs/lema.c10.r50.v3.log &

# nohup python train.py --config configs/lema.c10.yaml --gpu 1 --model resnet18 --epochs 350 --epochs_lin 100 --save_path lema.c10.r18.v3.pth > logs/lema.c10.r18.v3.log &

# nohup python train.py --config configs/lema.c100.yaml --gpu 2 --model resnet50 --epochs 350 --epochs_lin 100 --save_path lema.c100.r50.v3.pth > logs/lema.c100.r50.v3.log &

# nohup python train.py --config configs/lema.c100.yaml --gpu 1 --model resnet18 --epochs 350 --epochs_lin 100 --save_path lema.c100.r18.v3.pth > logs/lema.c100.r18.v3.log &


#### Running for modification in CARL

# nohup python train.py --config configs/carl.c10.yaml --gpu 6 --model resnet50 --epochs 1000 --epochs_lin 100 --save_path carl.c10.r50.ode5.v2.pth --ode_steps 5 > logs/carl.c10.r50.ode5.v2.log &

# nohup python train.py --config configs/carl.c10.yaml --gpu 7 --model resnet18 --epochs 1000 --epochs_lin 100 --save_path carl.c10.r18.ode5.v2.pth --ode_steps 5 > logs/carl.c10.r18.ode5.v2.log &

# nohup python train.py --config configs/carl.c100.yaml --gpu 5 --model resnet50 --epochs 1000 --epochs_lin 100 --save_path carl.c100.r50.ode5.v2.pth --ode_steps 5 > logs/carl.c100.r50.ode5.v2.log &

# nohup python train.py --config configs/carl.c100.yaml --gpu 6 --model resnet18 --epochs 1000 --epochs_lin 100 --save_path carl.c100.r18.ode5.v2.pth --ode_steps 5 > logs/carl.c100.r18.ode5.v2.log &


##### DAiLEMa

# nohup python train.py --config configs/dailema.c10.yaml --gpu 3 --model resnet50 --epochs 350 --epochs_lin 100 --vae_out 256 --save_path dailema.c10.r50.pth > logs/dailema.c10.r50.log &

# nohup python train.py --config configs/dailema.c10.yaml --gpu 1 --model resnet18 --epochs 350 --epochs_lin 100 --vae_out 256 --save_path dailema.c10.r18.pth > logs/dailema.c10.r18.log &

# nohup python train.py --config configs/dailema.c100.yaml --gpu 4 --model resnet50 --epochs 350 --epochs_lin 100 --vae_out 256 --save_path dailema.c100.r50.pth > logs/dailema.c100.r50.log &

# nohup python train.py --config configs/dailema.c100.yaml --gpu 1 --model resnet18 --epochs 350 --epochs_lin 100 --vae_out 256 --save_path dailema.c100.r18.pth > logs/dailema.c100.r18.log &

# nohup python train.py --config configs/dailema.c10.yaml --gpu 5 --model resnet50 --epochs 500 --epochs_lin 100 --vae_out 256 --save_path dailema.c10.r50.e500.pth > logs/dailema.c10.r50.e500.log &

# nohup python train.py --config configs/dailema.c10.yaml --gpu 2 --model resnet18 --epochs 500 --epochs_lin 100 --vae_out 256 --save_path dailema.c10.r18.e500.pth > logs/dailema.c10.r18.e500.log &

# nohup python train.py --config configs/dailema.c100.yaml --gpu 6 --model resnet50 --epochs 500 --epochs_lin 100 --vae_out 256 --save_path dailema.c100.r50.e500.pth > logs/dailema.c100.r50.e500.log &

# nohup python train.py --config configs/dailema.c100.yaml --gpu 2 --model resnet18 --epochs 500 --epochs_lin 100 --vae_out 256 --save_path dailema.c100.r18.e500.pth > logs/dailema.c100.r18.e500.log &


# nohup python train.py --config configs/scalre.c10.yaml --gpu 0 --model resnet50 --epochs 350 --epochs_lin 100 --save_path scalre.c10.r50.pth --net_type energy > logs/scalre.c10.r50.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 1 --model resnet18 --epochs 350 --epochs_lin 100 --save_path scalre.c10.r18.pth --net_type energy > logs/scalre.c10.r18.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 3 --model resnet50 --epochs 350 --epochs_lin 100 --save_path scalre.c100.r50.pth --net_type energy > logs/scalre.c100.r50.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 1 --model resnet18 --epochs 350 --epochs_lin 100 --save_path scalre.c100.r18.pth --net_type energy > logs/scalre.c100.r18.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 4 --model resnet50 --epochs 350 --epochs_lin 100 --save_path scalre.c10.r50.sc.pth --net_type score > logs/scalre.c10.r50.sc.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 6 --model resnet18 --epochs 350 --epochs_lin 100 --save_path scalre.c10.r18.sc.pth --net_type score > logs/scalre.c10.r18.sc.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 5 --model resnet50 --epochs 350 --epochs_lin 100 --save_path scalre.c100.r50.sc.pth --net_type score > logs/scalre.c100.r50.sc.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 6 --model resnet18 --epochs 350 --epochs_lin 100 --save_path scalre.c100.r18.sc.pth --net_type score > logs/scalre.c100.r18.sc.log &


# nohup python train.py --config configs/scalre.c10.yaml --gpu 0 --model resnet50 --epochs 350 --epochs_lin 100 --save_path scalre.c10.r50.ls.pth --langevin_steps 10 --net_type energy > logs/scalre.c10.r50.ls.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 1 --model resnet18 --epochs 350 --epochs_lin 100 --save_path scalre.c10.r18.ls.pth --langevin_steps 10 --net_type energy > logs/scalre.c10.r18.ls.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 3 --model resnet50 --epochs 350 --epochs_lin 100 --save_path scalre.c100.r50.ls.pth --langevin_steps 10 --net_type energy > logs/scalre.c100.r50.ls.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 1 --model resnet18 --epochs 350 --epochs_lin 100 --save_path scalre.c100.r18.ls.pth --langevin_steps 10 --net_type energy > logs/scalre.c100.r18.ls.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 4 --model resnet50 --epochs 350 --epochs_lin 100 --save_path scalre.c10.r50.sc.ls.pth --langevin_steps 10 --net_type score > logs/scalre.c10.r50.sc.ls.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 6 --model resnet18 --epochs 350 --epochs_lin 100 --save_path scalre.c10.r18.sc.ls.pth --langevin_steps 10 --net_type score > logs/scalre.c10.r18.sc.ls.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 5 --model resnet50 --epochs 350 --epochs_lin 100 --save_path scalre.c100.r50.sc.ls.pth --langevin_steps 10 --net_type score > logs/scalre.c100.r50.sc.ls.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 6 --model resnet18 --epochs 350 --epochs_lin 100 --save_path scalre.c100.r18.sc.ls.pth --langevin_steps 10 --net_type score > logs/scalre.c100.r18.sc.ls.log &


# nohup python train.py --config configs/scalre.c10.yaml --gpu 0 --model resnet50 --epochs 350 --epochs_lin 100 --save_path scalre.c10.r50.sc.we50.pth --warmup_epochs 50 --net_type score > logs/scalre.c10.r50.sc.we50.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 0 --model resnet18 --epochs 350 --epochs_lin 100 --save_path scalre.c10.r18.sc.we50.pth --warmup_epochs 50 --net_type score > logs/scalre.c10.r18.sc.we50.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 1 --model resnet50 --epochs 350 --epochs_lin 100 --save_path scalre.c100.r50.sc.we50.pth --warmup_epochs 50 --net_type score > logs/scalre.c100.r50.sc.we50.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 1 --model resnet18 --epochs 350 --epochs_lin 100 --save_path scalre.c100.r18.sc.we50.pth --warmup_epochs 50 --net_type score > logs/scalre.c100.r18.sc.we50.log &

nohup python train.py --config configs/lema.c10.yaml --gpu 0 --model resnet50 --save_path lema.c10.r50.v3.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear > logs/lema.c10.r50.v3.e800.lin.log &

nohup python train.py --config configs/lema.c10.yaml --gpu 0 --model resnet18 --save_path lema.c10.r18.v3.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear > logs/lema.c10.r18.v3.e800.lin.log &

nohup python train.py --config configs/lema.c100.yaml --gpu 1 --model resnet50 --save_path lema.c100.r50.v3.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear > logs/lema.c100.r50.v3.e800.lin.log &

nohup python train.py --config configs/lema.c100.yaml --gpu 1 --model resnet18 --save_path lema.c100.r18.v3.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear > logs/lema.c100.r18.v3.e800.lin.log &

nohup python train.py --config configs/scalre.c10.yaml --gpu 2 --model resnet50 --save_path scalre.c10.r50.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type energy > logs/scalre.c10.r50.e800.lin.log &

nohup python train.py --config configs/scalre.c10.yaml --gpu 2 --model resnet18 --save_path scalre.c10.r18.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type energy > logs/scalre.c10.r18.e800.lin.log &

nohup python train.py --config configs/scalre.c100.yaml --gpu 3 --model resnet50 --save_path scalre.c100.r50.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type energy > logs/scalre.c100.r50.e800.lin.log &

nohup python train.py --config configs/scalre.c100.yaml --gpu 3 --model resnet18 --save_path scalre.c100.r18.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type energy > logs/scalre.c100.r18.e800.lin.log &

nohup python train.py --config configs/scalre.c10.yaml --gpu 4 --model resnet50 --save_path scalre.c10.r50.sc.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type score > logs/scalre.c10.r50.sc.e800.lin.log &

nohup python train.py --config configs/scalre.c10.yaml --gpu 4 --model resnet18 --save_path scalre.c10.r18.sc.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type score > logs/scalre.c10.r18.sc.e800.lin.log &

nohup python train.py --config configs/scalre.c100.yaml --gpu 5 --model resnet50 --save_path scalre.c100.r50.sc.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type score > logs/scalre.c100.r50.sc.e800.lin.log &

nohup python train.py --config configs/scalre.c100.yaml --gpu 5 --model resnet18 --save_path scalre.c100.r18.sc.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type score > logs/scalre.c100.r18.sc.e800.lin.log &
