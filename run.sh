#!/bin/bash 

########################### 800 epochs experiments ##############################
#################################################################################

# Experiment for simsiam 

# nohup python train.py --config configs/simsiam.c10.yaml --gpu 6 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --save_path simsiam.c10.r18.e800.pth > logs/simsiam.c10.r18.e800.log &

# nohup python train.py --config configs/simsiam.c100.yaml --gpu 7 --model resnet18 --epochs 800 --epochs_lin 100 --lr 0.08 --linear_lr 0.1 --mlp_type linear --save_path simsiam.c100.r18.e800.pth > logs/simsiam.c100.r18.e800.log &

# nohup python train.py --config configs/simsiam.c10.yaml --gpu 0 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --save_path simsiam.c10.r50.e800.pth > logs/simsiam.c10.r50.e800.log &

# nohup python train.py --config configs/simsiam.c100.yaml --gpu 2 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --save_path simsiam.c100.r50.e800.pth > logs/simsiam.c100.r50.e800.log &

# Experiment for simclr

# nohup python train.py --config configs/simclr.c10.yaml --mlp_type linear --gpu 5 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --save_path simclr.c10.r18.e800.pth > logs/simclr.c10.r18.e800.log &

# nohup python train.py --config configs/simclr.c100.yaml --mlp_type linear --gpu 5 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --save_path simclr.c100.r18.e800.pth > logs/simclr.c100.r18.e800.log &

# nohup python train.py --config configs/simclr.c10.yaml --mlp_type linear --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --save_path simclr.c10.r50.e800.pth > logs/simclr.c10.r50.e800.log &

# nohup python train.py --config configs/simclr.c100.yaml --mlp_type linear --gpu 7 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --save_path simclr.c100.r50.e800.pth > logs/simclr.c100.r50.e800.log &


# experiments for barlow twins

# nohup python train.py --config configs/barlow_twins.c10.yaml --gpu 2 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --save_path bt.c10.r18.e800.pth > logs/bt.c10.r18.e800.log &

# nohup python train.py --config configs/barlow_twins.c100.yaml --gpu 2 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --save_path bt.c100.r18.e800.pth > logs/bt.c100.r18.e800.log &

# nohup python train.py --config configs/barlow_twins.c10.yaml --gpu 3 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --save_path bt.c10.r50.e800.pth > logs/bt.c10.r50.e800.log &

# nohup python train.py --config configs/barlow_twins.c100.yaml --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --save_path bt.c100.r50.e800.pth > logs/bt.c100.r50.e800.log &

# experiments for byol 

# nohup python train.py --config configs/byol.c10.yaml --gpu 6 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --save_path byol.c10.r18.e800.pth > logs/byol.c10.r18.e800.log &

# nohup python train.py --config configs/byol.c100.yaml --gpu 7 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --save_path byol.c100.r18.e800.pth > logs/byol.c100.r18.e800.log &

# nohup python train.py --config configs/byol.c10.yaml --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --save_path byol.c10.r50.e800.pth > logs/byol.c10.r50.e800.log &

# nohup python train.py --config configs/byol.c100.yaml --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --lr 0.2 --save_path byol.c100.r50.e800.pth > logs/byol.c100.r50.e800.log &

# experiments for vicreg 

# nohup python train.py --config configs/vicreg.c10.yaml --gpu 3 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --save_path vicreg.c10.r18.e800.pth > logs/vicreg.c10.r18.e800.log &

# nohup python train.py --config configs/vicreg.c100.yaml --gpu 2 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --save_path vicreg.c100.r18.e800.pth > logs/vicreg.c100.r18.e800.log &

# nohup python train.py --config configs/vicreg.c10.yaml --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --save_path vicreg.c10.r50.e800.pth > logs/vicreg.c10.r50.e800.log &

# nohup python train.py --config configs/vicreg.c100.yaml --gpu 3 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --save_path vicreg.c100.r50.e800.pth > logs/vicreg.c100.r50.e800.log &


################################# 800 epochs experiments #############################
######################################################################################


# nohup python train.py --config configs/scalre.c10.yaml --gpu 2 --model resnet50 --save_path scalre.c10.r50.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type energy > logs/scalre.c10.r50.e800.lin.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 2 --model resnet18 --save_path scalre.c10.r18.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type energy > logs/scalre.c10.r18.e800.lin.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 3 --model resnet50 --save_path scalre.c100.r50.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type energy > logs/scalre.c100.r50.e800.lin.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 3 --model resnet18 --save_path scalre.c100.r18.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type energy > logs/scalre.c100.r18.e800.lin.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 4 --model resnet50 --save_path scalre.c10.r50.sc.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type score > logs/scalre.c10.r50.sc.e800.lin.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 4 --model resnet18 --save_path scalre.c10.r18.sc.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type score > logs/scalre.c10.r18.sc.e800.lin.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 5 --model resnet50 --save_path scalre.c100.r50.sc.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type score > logs/scalre.c100.r50.sc.e800.lin.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 5 --model resnet18 --save_path scalre.c100.r18.sc.e800.lin.pth --epochs 800 --epochs_lin 100 --mlp_type linear --net_type score > logs/scalre.c100.r18.sc.e800.lin.log &


# nohup python train.py --config configs/simsiam.sc.c10.yaml --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type score --save_path simsiam.sc.c10.r18.e800.pth > logs/simsiam.sc.c10.r18.e800.log &

# nohup python train.py --config configs/simsiam.sc.c100.yaml --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --lr 0.08 --linear_lr 0.1 --mlp_type linear --net_type score --save_path simsiam.sc.c100.r18.e800.pth > logs/simsiam.sc.c100.r18.e800.log &

# nohup python train.py --config configs/simsiam.sc.c10.yaml --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type score --save_path simsiam.sc.c10.r50.e800.pth > logs/simsiam.sc.c10.r50.e800.log &

# nohup python train.py --config configs/simsiam.sc.c100.yaml --gpu 5 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type score --save_path simsiam.sc.c100.r50.e800.pth > logs/simsiam.sc.c100.r50.e800.log &

# nohup python train.py --config configs/simsiam.sc.c10.yaml --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type energy --save_path simsiam.en.c10.r18.e800.pth > logs/simsiam.en.c10.r18.e800.log &

# nohup python train.py --config configs/simsiam.sc.c100.yaml --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --lr 0.08 --linear_lr 0.1 --mlp_type linear --net_type energy --save_path simsiam.en.c100.r18.e800.pth > logs/simsiam.en.c100.r18.e800.log &

# nohup python train.py --config configs/simsiam.sc.c10.yaml --gpu 0 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type energy --save_path simsiam.en.c10.r50.e800.pth > logs/simsiam.en.c10.r50.e800.log &

# nohup python train.py --config configs/simsiam.sc.c100.yaml --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type energy --save_path simsiam.en.c100.r50.e800.pth > logs/simsiam.en.c100.r50.e800.log &


# nohup python train.py --config configs/bt.sc.c10.yaml --gpu 2 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type score --save_path bt.sc.c10.r18.e800.pth > logs/bt.sc.c10.r18.e800.log &

# nohup python train.py --config configs/bt.sc.c100.yaml --gpu 2 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type score --save_path bt.sc.c100.r18.e800.pth > logs/bt.sc.c100.r18.e800.log &

# nohup python train.py --config configs/bt.sc.c10.yaml --gpu 3 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type score --save_path bt.sc.c10.r50.e800.pth > logs/bt.sc.c10.r50.e800.log &

# nohup python train.py --config configs/bt.sc.c100.yaml --gpu 4 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type score --save_path bt.sc.c100.r50.e800.pth > logs/bt.sc.c100.r50.e800.log &

# nohup python train.py --config configs/bt.sc.c10.yaml --gpu 2 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type energy --save_path bt.en.c10.r18.e800.pth > logs/bt.en.c10.r18.e800.log &

# nohup python train.py --config configs/bt.sc.c100.yaml --gpu 2 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type energy --save_path bt.en.c100.r18.e800.pth > logs/bt.en.c100.r18.e800.log &

# nohup python train.py --config configs/bt.sc.c10.yaml --gpu 3 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type energy --save_path bt.en.c10.r50.e800.pth > logs/bt.en.c10.r50.e800.log &

# nohup python train.py --config configs/bt.sc.c100.yaml --gpu 4 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type energy --save_path bt.en.c100.r50.e800.pth > logs/bt.en.c100.r50.e800.log &


# nohup python train.py --config configs/byol.sc.c10.yaml --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type energy --save_path byol.en.c10.r18.e800.pth > logs/byol.en.c10.r18.e800.log &

# nohup python train.py --config configs/byol.sc.c100.yaml --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type energy --save_path byol.en.c100.r18.e800.pth > logs/byol.en.c100.r18.e800.log &

# nohup python train.py --config configs/byol.sc.c10.yaml --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type energy --save_path byol.en.c10.r50.e800.pth > logs/byol.en.c10.r50.e800.log &

# nohup python train.py --config configs/byol.sc.c100.yaml --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type energy --lr 0.2 --save_path byol.en.c100.r50.e800.pth > logs/byol.en.c100.r50.e800.log &

# nohup python train.py --config configs/byol.sc.c10.yaml --gpu 4 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type score --save_path byol.sc.c10.r18.e800.pth > logs/byol.sc.c10.r18.e800.log &

# nohup python train.py --config configs/byol.sc.c100.yaml --gpu 5 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type score --save_path byol.sc.c100.r18.e800.pth > logs/byol.sc.c100.r18.e800.log &

# nohup python train.py --config configs/byol.sc.c10.yaml --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type score --save_path byol.sc.c10.r50.e800.pth > logs/byol.sc.c10.r50.e800.log &

# nohup python train.py --config configs/byol.sc.c100.yaml --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --net_type score --lr 0.2 --save_path byol.sc.c100.r50.e800.pth > logs/byol.sc.c100.r50.e800.log &


# nohup python train.py --config configs/vicreg.sc.c10.yaml --gpu 4 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --net_type score --save_path vicreg.c10.r18.sc.e800.pth > logs/vicreg.c10.r18.sc.e800.log &

# nohup python train.py --config configs/vicreg.sc.c100.yaml --gpu 5 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --net_type score --save_path vicreg.c100.r18.sc.e800.pth > logs/vicreg.c100.r18.sc.e800.log &

# nohup python train.py --config configs/vicreg.sc.c10.yaml --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --net_type score --save_path vicreg.c10.r50.sc.e800.pth > logs/vicreg.c10.r50.sc.e800.log &

# nohup python train.py --config configs/vicreg.sc.c100.yaml --gpu 7 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --net_type score --save_path vicreg.c100.r50.sc.e800.pth > logs/vicreg.c100.r50.sc.e800.log &

# nohup python train.py --config configs/vicreg.sc.c10.yaml --gpu 6 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --net_type energy --save_path vicreg.c10.r18.en.e800.pth > logs/vicreg.c10.r18.en.e800.log &

# nohup python train.py --config configs/vicreg.sc.c100.yaml --gpu 7 --model resnet18 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --net_type energy --save_path vicreg.c100.r18.en.e800.pth > logs/vicreg.c100.r18.en.e800.log &

# nohup python train.py --config configs/vicreg.sc.c10.yaml --gpu 4 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --net_type energy --save_path vicreg.c10.r50.en.e800.pth > logs/vicreg.c10.r50.en.e800.log &

# nohup python train.py --config configs/vicreg.sc.c100.yaml --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --linear_lr 0.1 --mlp_type linear --opt LARS --net_type energy --save_path vicreg.c100.r50.en.e800.pth > logs/vicreg.c100.r50.en.e800.log &


######################## 800 epochs updated experiments ####################### 
###############################################################################

# barlow twins experiments 

# nohup python train.py --config configs/bt.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --save_path bt.c10.r18.pth > logs/bt.c10.r18.log &

# nohup python train.py --config configs/bt.yaml --dataset cifar100 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100  --save_path bt.c100.r18.pth > logs/bt.c100.r18.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type energy --save_path bt.en.c10.r18.pth > logs/bt.en.c10.r18.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type score --save_path bt.sc.c10.r18.pth > logs/bt.sc.c10.r18.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar100 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type energy --save_path bt.en.c100.r18.pth > logs/bt.en.c100.r18.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar100 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type score --save_path bt.sc.c100.r18.pth > logs/bt.sc.c100.r18.log &


# vicreg experiments 

# nohup python train.py --config configs/vicreg.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --save_path vicreg.c10.r18.pth > logs/vicreg.c10.r18.log &

# nohup python train.py --config configs/vicreg.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --save_path vicreg.c100.r18.pth > logs/vicreg.c100.r18.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --net_type energy --save_path vicreg.en.c10.r18.pth > logs/vicreg.en.c10.r18.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --net_type score --save_path vicreg.sc.c10.r18.pth > logs/vicreg.sc.c10.r18.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --net_type energy --save_path vicreg.en.c100.r18.pth > logs/vicreg.en.c100.r18.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --net_type score --save_path vicreg.sc.c100.r18.pth > logs/vicreg.sc.c100.r18.log &

# simsiam experiments  

# nohup python train.py --config configs/simsiam.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --save_path simsiam.c10.r18.pth > logs/simsiam.c10.r18.log &

# nohup python train.py --config configs/simsiam.yaml --dataset cifar100 --gpu 0 --model resnet18 --lr 0.08 --epochs 800 --epochs_lin 100  --save_path simsiam.c100.r18.pth > logs/simsiam.c100.r18.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --net_type energy --save_path simsiam.en.c10.r18.pth > logs/simsiam.en.c10.r18.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --net_type score --save_path simsiam.sc.c10.r18.pth > logs/simsiam.sc.c10.r18.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --lr 0.08 --net_type energy --save_path simsiam.en.c100.r18.pth > logs/simsiam.en.c100.r18.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --lr 0.08 --net_type score --save_path simsiam.sc.c100.r18.pth > logs/simsiam.sc.c100.r18.log &


# simclr experiments 

# nohup python train.py --config configs/simclr.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --save_path simclr.c10.r18.pth > logs/simclr.c10.r18.log &

# nohup python train.py --config configs/simclr.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --save_path simclr.c100.r18.pth > logs/simclr.c100.r18.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type energy --save_path scalre.en.c10.r18.pth > logs/scalre.en.c10.r18.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type score --save_path scalre.sc.c10.r18.pth > logs/scalre.sc.c10.r18.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar100 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type energy --save_path scalre.en.c100.r18.pth > logs/scalre.en.c100.r18.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar100 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type score --save_path scalre.sc.c100.r18.pth > logs/scalre.sc.c100.r18.log &


# byol experiments 

# nohup python train.py --config configs/byol.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --save_path byol.c10.r18.pth > logs/byol.c10.r18.log &

# nohup python train.py --config configs/byol.yaml --dataset cifar100 --gpu 1 --model resnet18 --wd 1e-4 --epochs 800 --epochs_lin 100 --save_path byol.c100.r18.pth > logs/byol.c100.r18.log &


######################## 800 epochs updated experiments (r50) ####################### 
###############################################################################

# barlow twins experiments 

# nohup python train.py --config configs/bt.yaml --dataset cifar10 --gpu 0 --model resnet50 --epochs 800 --epochs_lin 100 --save_path bt.c10.r50.pth > logs/bt.c10.r50.log &

# nohup python train.py --config configs/bt.yaml --dataset cifar100 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100  --save_path bt.c100.r50.pth > logs/bt.c100.r50.log &

nohup python train.py --config configs/bt.sc.yaml --dataset cifar10 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --net_type energy --save_path bt.en.c10.r50.pth > logs/bt.en.c10.r50.log &

nohup python train.py --config configs/bt.sc.yaml --dataset cifar10 --gpu 2 --model resnet50 --epochs 800 --epochs_lin 100 --net_type score --save_path bt.sc.c10.r50.pth > logs/bt.sc.c10.r50.log &

nohup python train.py --config configs/bt.sc.yaml --dataset cifar100 --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --net_type energy --save_path bt.en.c100.r50.pth > logs/bt.en.c100.r50.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar100 --gpu 0 --model resnet50 --epochs 800 --epochs_lin 100 --net_type score --save_path bt.sc.c100.r50.pth > logs/bt.sc.c100.r50.log &


# vicreg experiments 

# nohup python train.py --config configs/vicreg.yaml --dataset cifar10 --gpu 2 --model resnet50 --epochs 800 --epochs_lin 100 --opt LARS --save_path vicreg.c10.r50.pth > logs/vicreg.c10.r50.log &

# nohup python train.py --config configs/vicreg.yaml --dataset cifar100 --gpu 3 --model resnet50 --epochs 800 --epochs_lin 100 --opt LARS --save_path vicreg.c100.r50.pth > logs/vicreg.c100.r50.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar10 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --opt LARS --net_type energy --save_path vicreg.en.c10.r50.pth > logs/vicreg.en.c10.r50.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar10 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --opt LARS --net_type score --save_path vicreg.sc.c10.r50.pth > logs/vicreg.sc.c10.r50.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar100 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --opt LARS --net_type energy --save_path vicreg.en.c100.r50.pth > logs/vicreg.en.c100.r50.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar100 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --opt LARS --net_type score --save_path vicreg.sc.c100.r50.pth > logs/vicreg.sc.c100.r50.log &

# simsiam experiments  

# nohup python train.py --config configs/simsiam.yaml --dataset cifar10 --gpu 4 --model resnet50 --epochs 800 --epochs_lin 100 --save_path simsiam.c10.r50.pth > logs/simsiam.c10.r50.log &

# nohup python train.py --config configs/simsiam.yaml --dataset cifar100 --gpu 5 --model resnet50 --lr 0.08 --epochs 800 --epochs_lin 100  --save_path simsiam.c100.r50.pth > logs/simsiam.c100.r50.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar10 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --net_type energy --save_path simsiam.en.c10.r50.pth > logs/simsiam.en.c10.r50.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar10 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --net_type score --save_path simsiam.sc.c10.r50.pth > logs/simsiam.sc.c10.r50.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar100 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --lr 0.08 --net_type energy --save_path simsiam.en.c100.r50.pth > logs/simsiam.en.c100.r50.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar100 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --lr 0.08 --net_type score --save_path simsiam.sc.c100.r50.pth > logs/simsiam.sc.c100.r50.log &


# simclr experiments 

# nohup python train.py --config configs/simclr.yaml --dataset cifar10 --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --save_path simclr.c10.r50.pth > logs/simclr.c10.r50.log &

# nohup python train.py --config configs/simclr.yaml --dataset cifar100 --gpu 7 --model resnet50 --epochs 800 --epochs_lin 100 --save_path simclr.c100.r50.pth > logs/simclr.c100.r50.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar10 --gpu 0 --model resnet50 --epochs 800 --epochs_lin 100 --net_type energy --save_path scalre.en.c10.r50.pth > logs/scalre.en.c10.r50.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar10 --gpu 0 --model resnet50 --epochs 800 --epochs_lin 100 --net_type score --save_path scalre.sc.c10.r50.pth > logs/scalre.sc.c10.r50.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar100 --gpu 0 --model resnet50 --epochs 800 --epochs_lin 100 --net_type energy --save_path scalre.en.c100.r50.pth > logs/scalre.en.c100.r50.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar100 --gpu 0 --model resnet50 --epochs 800 --epochs_lin 100 --net_type score --save_path scalre.sc.c100.r50.pth > logs/scalre.sc.c100.r50.log &


# byol experiments 

# nohup python train.py --config configs/byol.yaml --dataset cifar10 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --save_path byol.c10.r50.pth > logs/byol.c10.r50.log &

# nohup python train.py --config configs/byol.yaml --dataset cifar100 --gpu 1 --model resnet50 --wd 1e-4 --epochs 800 --epochs_lin 100 --save_path byol.c100.r50.pth > logs/byol.c100.r50.log &