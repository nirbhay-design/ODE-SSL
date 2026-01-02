#!/bin/bash 

nohup python train.py --config configs/lema.c10.yaml --gpu 6 --model resnet50 --save_path lema.c10.r50.v3.pth --epochs 350 --epochs_lin 100 --test --tsne --knn --lreg > test_garbage/lema.c10.r50.v3.log &

nohup python train.py --config configs/lema.c10.yaml --gpu 6 --model resnet18 --save_path lema.c10.r18.v3.pth --epochs 350 --epochs_lin 100 --test --tsne --knn --lreg > test_garbage/lema.c10.r18.v3.log &

nohup python train.py --config configs/lema.c100.yaml --gpu 6 --model resnet50 --save_path lema.c100.r50.v3.pth --epochs 350 --epochs_lin 100 --test --tsne --knn --lreg > test_garbage/lema.c100.r50.v3.log &

nohup python train.py --config configs/lema.c100.yaml --gpu 6 --model resnet18 --save_path lema.c100.r18.v3.pth --epochs 350 --epochs_lin 100 --test --tsne --knn --lreg > test_garbage/lema.c100.r18.v3.log &

nohup python train.py --config configs/scalre.c10.yaml --gpu 6 --model resnet50 --save_path scalre.c10.r50.pth --epochs 350 --epochs_lin 100 --net_type energy --test --tsne --knn --lreg > test_garbage/scalre.c10.r50.log &

nohup python train.py --config configs/scalre.c10.yaml --gpu 6 --model resnet18 --save_path scalre.c10.r18.pth --epochs 350 --epochs_lin 100 --net_type energy --test --tsne --knn --lreg > test_garbage/scalre.c10.r18.log &

nohup python train.py --config configs/scalre.c100.yaml --gpu 7 --model resnet50 --save_path scalre.c100.r50.pth --epochs 350 --epochs_lin 100 --net_type energy --test --tsne --knn --lreg > test_garbage/scalre.c100.r50.log &

nohup python train.py --config configs/scalre.c100.yaml --gpu 7 --model resnet18 --save_path scalre.c100.r18.pth --epochs 350 --epochs_lin 100 --net_type energy --test --tsne --knn --lreg > test_garbage/scalre.c100.r18.log &

nohup python train.py --config configs/scalre.c10.yaml --gpu 7 --model resnet50 --save_path scalre.c10.r50.sc.pth --epochs 350 --epochs_lin 100 --net_type score --test --tsne --knn --lreg > test_garbage/scalre.c10.r50.sc.log &

nohup python train.py --config configs/scalre.c10.yaml --gpu 7 --model resnet18 --save_path scalre.c10.r18.sc.pth --epochs 350 --epochs_lin 100 --net_type score --test --tsne --knn --lreg > test_garbage/scalre.c10.r18.sc.log &

nohup python train.py --config configs/scalre.c100.yaml --gpu 7 --model resnet50 --save_path scalre.c100.r50.sc.pth --epochs 350 --epochs_lin 100 --net_type score --test --tsne --knn --lreg > test_garbage/scalre.c100.r50.log &

nohup python train.py --config configs/scalre.c100.yaml --gpu 7 --model resnet18 --save_path scalre.c100.r18.sc.pth --epochs 350 --epochs_lin 100 --net_type score --test --tsne --knn --lreg > test_garbage/scalre.c100.r18.log &
