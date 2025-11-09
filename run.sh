#!/bin/bash 

nohup python train.py --config configs/nodel.c10.yaml --gpu 0 --model resnet50 --epochs 600 --epochs_lin 100 --save_path nodel.c10.r50.pth > logs/nodel.c10.r50.log &

nohup python train.py --config configs/nodel.c10.yaml --gpu 0 --model resnet18 --epochs 600 --epochs_lin 100 --save_path nodel.c10.r18.pth > logs/nodel.c10.r18.log &

nohup python train.py --config configs/carl.c10.yaml --gpu 1 --model resnet50 --epochs 1000 --epochs_lin 100 --save_path carl.c10.r50.pth > logs/carl.c10.r50.log &

nohup python train.py --config configs/carl.c10.yaml --gpu 0 --model resnet18 --epochs 1000 --epochs_lin 100 --save_path carl.c10.r18.pth > logs/carl.c10.r18.log &