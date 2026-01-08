#!/bin/bash 

# nohup python train.py --config configs/lema.c10.yaml --gpu 6 --model resnet50 --save_path lema.c10.r50.v3.pth --epochs 350 --epochs_lin 100 --test --tsne --knn --lreg > test_garbage/lema.c10.r50.v3.log &

# nohup python train.py --config configs/lema.c10.yaml --gpu 6 --model resnet18 --save_path lema.c10.r18.v3.pth --epochs 350 --epochs_lin 100 --test --tsne --knn --lreg > test_garbage/lema.c10.r18.v3.log &

# nohup python train.py --config configs/lema.c100.yaml --gpu 6 --model resnet50 --save_path lema.c100.r50.v3.pth --epochs 350 --epochs_lin 100 --test --tsne --knn --lreg > test_garbage/lema.c100.r50.v3.log &

# nohup python train.py --config configs/lema.c100.yaml --gpu 6 --model resnet18 --save_path lema.c100.r18.v3.pth --epochs 350 --epochs_lin 100 --test --tsne --knn --lreg > test_garbage/lema.c100.r18.v3.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 0 --model resnet50 --save_path scalre.c10.r50.pth --epochs 350 --epochs_lin 100 --net_type energy --test --tsne --knn --lreg > test_garbage/scalre.c10.r50.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 0 --model resnet18 --save_path scalre.c10.r18.pth --epochs 350 --epochs_lin 100 --net_type energy --test --tsne --knn --lreg > test_garbage/scalre.c10.r18.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 7 --model resnet50 --save_path scalre.c100.r50.pth --epochs 350 --epochs_lin 100 --net_type energy --test --tsne --knn --lreg > test_garbage/scalre.c100.r50.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 1 --model resnet18 --save_path scalre.c100.r18.pth --epochs 350 --epochs_lin 100 --net_type energy --test --tsne --knn --lreg > test_garbage/scalre.c100.r18.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 7 --model resnet50 --save_path scalre.c10.r50.sc.pth --epochs 350 --epochs_lin 100 --net_type score --test --tsne --knn --lreg > test_garbage/scalre.c10.r50.sc.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 7 --model resnet18 --save_path scalre.c10.r18.sc.pth --epochs 350 --epochs_lin 100 --net_type score --test --tsne --knn --lreg > test_garbage/scalre.c10.r18.sc.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 7 --model resnet50 --save_path scalre.c100.r50.sc.pth --epochs 350 --epochs_lin 100 --net_type score --test --tsne --knn --lreg > test_garbage/scalre.c100.r50.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 7 --model resnet18 --save_path scalre.c100.r18.sc.pth --epochs 350 --epochs_lin 100 --net_type score --test --tsne --knn --lreg > test_garbage/scalre.c100.r18.log &


# nohup python train.py --config configs/lema.c10.yaml --gpu 0 --model resnet50 --save_path lema.c10.r50.v3.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 > logs/lema.c10.r50.v3.e800.lin0.1.log &

# nohup python train.py --config configs/lema.c10.yaml --gpu 0 --model resnet18 --save_path lema.c10.r18.v3.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 > logs/lema.c10.r18.v3.e800.lin0.1.log &

# nohup python train.py --config configs/lema.c100.yaml --gpu 1 --model resnet50 --save_path lema.c100.r50.v3.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 > logs/lema.c100.r50.v3.e800.lin0.1.log &

# nohup python train.py --config configs/lema.c100.yaml --gpu 1 --model resnet18 --save_path lema.c100.r18.v3.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 > logs/lema.c100.r18.v3.e800.lin0.1.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 2 --model resnet50 --save_path scalre.c10.r50.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type energy > logs/scalre.c10.r50.e800.lin0.1.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 3 --model resnet18 --save_path scalre.c10.r18.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type energy > logs/scalre.c10.r18.e800.lin0.1.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 4 --model resnet50 --save_path scalre.c100.r50.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type energy > logs/scalre.c100.r50.e800.lin0.1.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 3 --model resnet18 --save_path scalre.c100.r18.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type energy > logs/scalre.c100.r18.e800.lin0.1.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 1 --model resnet50 --save_path scalre.c10.r50.sc.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type score > logs/scalre.c10.r50.sc.e800.lin0.1.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 0 --model resnet18 --save_path scalre.c10.r18.sc.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type score > logs/scalre.c10.r18.sc.e800.lin0.1.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 7 --model resnet50 --save_path scalre.c100.r50.sc.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type score > logs/scalre.c100.r50.sc.e800.lin0.1.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 0 --model resnet18 --save_path scalre.c100.r18.sc.e800.lin0.1.pth --epochs 800 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type score > logs/scalre.c100.r18.sc.e800.lin0.1.log &


nohup python train.py --config configs/lema.c10.yaml --gpu 0 --model resnet50 --save_path lema.c10.r50.v3.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 > logs/lema.c10.r50.v3.e350.lin0.1.log &

nohup python train.py --config configs/lema.c10.yaml --gpu 0 --model resnet18 --save_path lema.c10.r18.v3.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 > logs/lema.c10.r18.v3.e350.lin0.1.log &

nohup python train.py --config configs/lema.c100.yaml --gpu 1 --model resnet50 --save_path lema.c100.r50.v3.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 > logs/lema.c100.r50.v3.e350.lin0.1.log &

nohup python train.py --config configs/lema.c100.yaml --gpu 1 --model resnet18 --save_path lema.c100.r18.v3.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 > logs/lema.c100.r18.v3.e350.lin0.1.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 0 --model resnet50 --save_path scalre.c10.r50.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type energy > logs/scalre.c10.r50.e350.lin0.1.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 1 --model resnet18 --save_path scalre.c10.r18.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type energy > logs/scalre.c10.r18.e350.lin0.1.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 2 --model resnet50 --save_path scalre.c100.r50.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type energy > logs/scalre.c100.r50.e350.lin0.1.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 1 --model resnet18 --save_path scalre.c100.r18.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type energy > logs/scalre.c100.r18.e350.lin0.1.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 3 --model resnet50 --save_path scalre.c10.r50.sc.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type score > logs/scalre.c10.r50.sc.e350.lin0.1.log &

# nohup python train.py --config configs/scalre.c10.yaml --gpu 4 --model resnet18 --save_path scalre.c10.r18.sc.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type score > logs/scalre.c10.r18.sc.e350.lin0.1.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 5 --model resnet50 --save_path scalre.c100.r50.sc.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type score > logs/scalre.c100.r50.sc.e350.lin0.1.log &

# nohup python train.py --config configs/scalre.c100.yaml --gpu 4 --model resnet18 --save_path scalre.c100.r18.sc.e350.lin0.1.pth --epochs 350 --epochs_lin 100 --mlp_type linear --linear_lr 0.1 --net_type score > logs/scalre.c100.r18.sc.e350.lin0.1.log &
