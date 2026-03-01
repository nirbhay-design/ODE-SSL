#!/bin/bash 

######################## 800 epochs updated experiments (r18) ####################### 
###############################################################################

# barlow twins experiments 

# nohup python train.py --config configs/bt.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --save_path bt.c10.r18.pth > logs/bt.c10.r18.log &

# nohup python train.py --config configs/bt.yaml --dataset cifar100 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100  --save_path bt.c100.r18.pth > logs/bt.c100.r18.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type energy --save_path bt.en.c10.r18.pth > logs/bt.en.c10.r18.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type score --save_path bt.sc.c10.r18.pth > logs/bt.sc.c10.r18.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar100 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type energy --save_path bt.en.c100.r18.pth > logs/bt.en.c100.r18.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar100 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type score --save_path bt.sc.c100.r18.pth > logs/bt.sc.c100.r18.log &

# nohup python train.py --config configs/bt.yaml --dataset timg --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --lr 0.2 --save_path bt.timg.r18.pth > logs/bt.timg.r18.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset timg --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --lr 0.2 --net_type energy --save_path bt.en.timg.r18.pth > logs/bt.en.timg.r18.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset timg --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --lr 0.2 --net_type score --save_path bt.sc.timg.r18.pth > logs/bt.sc.timg.r18.log &


# vicreg experiments 

# nohup python train.py --config configs/vicreg.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --save_path vicreg.c10.r18.pth > logs/vicreg.c10.r18.log &

# nohup python train.py --config configs/vicreg.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --save_path vicreg.c100.r18.pth > logs/vicreg.c100.r18.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --net_type energy --save_path vicreg.en.c10.r18.pth > logs/vicreg.en.c10.r18.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --net_type score --save_path vicreg.sc.c10.r18.pth > logs/vicreg.sc.c10.r18.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --net_type energy --save_path vicreg.en.c100.r18.pth > logs/vicreg.en.c100.r18.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --net_type score --save_path vicreg.sc.c100.r18.pth > logs/vicreg.sc.c100.r18.log &

# nohup python train.py --config configs/vicreg.yaml --dataset timg --lr 0.2 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --save_path vicreg.timg.r18.pth > logs/vicreg.timg.r18.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset timg --lr 0.2 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --net_type energy --save_path vicreg.en.timg.r18.pth > logs/vicreg.en.timg.r18.log &

# nohup python train.py --config configs/vicreg.sc.yaml --dataset timg --lr 0.2 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --opt LARS --net_type score --save_path vicreg.sc.timg.r18.pth > logs/vicreg.sc.timg.r18.log &


# simsiam experiments  

# nohup python train.py --config configs/simsiam.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --save_path simsiam.c10.r18.pth > logs/simsiam.c10.r18.log &

# nohup python train.py --config configs/simsiam.yaml --dataset cifar100 --gpu 0 --model resnet18 --lr 0.08 --epochs 800 --epochs_lin 100  --save_path simsiam.c100.r18.pth > logs/simsiam.c100.r18.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --net_type energy --save_path simsiam.en.c10.r18.pth > logs/simsiam.en.c10.r18.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --net_type score --save_path simsiam.sc.c10.r18.pth > logs/simsiam.sc.c10.r18.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --lr 0.08 --net_type energy --save_path simsiam.en.c100.r18.pth > logs/simsiam.en.c100.r18.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --lr 0.08 --net_type score --save_path simsiam.sc.c100.r18.pth > logs/simsiam.sc.c100.r18.log &

# nohup python train.py --config configs/simsiam.yaml --dataset timg --gpu 2 --model resnet18 --lr 0.05 --epochs 800 --epochs_lin 100  --save_path simsiam.timg.r18.pth > logs/simsiam.timg.r18.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset timg --gpu 2 --model resnet18 --lr 0.05 --epochs 800 --epochs_lin 100 --net_type energy --save_path simsiam.en.timg.r18.pth > logs/simsiam.en.timg.r18.log &

# nohup python train.py --config configs/simsiam.sc.yaml --dataset timg --gpu 2 --model resnet18 --lr 0.05 --epochs 800 --epochs_lin 100 --net_type score --save_path simsiam.sc.timg.r18.pth > logs/simsiam.sc.timg.r18.log &


# simclr experiments 

# nohup python train.py --config configs/simclr.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --save_path simclr.c10.r18.pth > logs/simclr.c10.r18.log &

# nohup python train.py --config configs/simclr.yaml --dataset cifar100 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --save_path simclr.c100.r18.pth > logs/simclr.c100.r18.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type energy --save_path scalre.en.c10.r18.pth > logs/scalre.en.c10.r18.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar10 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type score --save_path scalre.sc.c10.r18.pth > logs/scalre.sc.c10.r18.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar100 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type energy --save_path scalre.en.c100.r18.pth > logs/scalre.en.c100.r18.log &

# nohup python train.py --config configs/scalre.yaml --dataset cifar100 --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type score --save_path scalre.sc.c100.r18.pth > logs/scalre.sc.c100.r18.log &

# nohup python train.py --config configs/simclr.yaml --dataset timg --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --save_path simclr.timg.r18.pth > logs/simclr.timg.r18.log &

# nohup python train.py --config configs/scalre.yaml --dataset timg --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type energy --save_path scalre.en.timg.r18.pth > logs/scalre.en.timg.r18.log &

# nohup python train.py --config configs/scalre.yaml --dataset timg --gpu 0 --model resnet18 --epochs 800 --epochs_lin 100 --net_type score --save_path scalre.sc.timg.r18.pth > logs/scalre.sc.timg.r18.log &

# byol experiments 

# nohup python train.py --config configs/byol.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --save_path byol.c10.r18.pth > logs/byol.c10.r18.log &

# nohup python train.py --config configs/byol.yaml --dataset cifar100 --gpu 1 --model resnet18 --wd 1e-4 --epochs 800 --epochs_lin 100 --save_path byol.c100.r18.pth > logs/byol.c100.r18.log &


######################## 800 epochs updated experiments (r50) ####################### 
###############################################################################

# barlow twins experiments 

# nohup python train.py --config configs/bt.yaml --dataset cifar10 --gpu 0 --model resnet50 --epochs 800 --epochs_lin 100 --save_path bt.c10.r50.pth > logs/bt.c10.r50.log &

# nohup python train.py --config configs/bt.yaml --dataset cifar100 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100  --save_path bt.c100.r50.pth > logs/bt.c100.r50.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar10 --gpu 1 --model resnet50 --epochs 800 --epochs_lin 100 --net_type energy --save_path bt.en.c10.r50.pth > logs/bt.en.c10.r50.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar10 --gpu 2 --model resnet50 --epochs 800 --epochs_lin 100 --net_type score --save_path bt.sc.c10.r50.pth > logs/bt.sc.c10.r50.log &

# nohup python train.py --config configs/bt.sc.yaml --dataset cifar100 --gpu 6 --model resnet50 --epochs 800 --epochs_lin 100 --net_type energy --save_path bt.en.c100.r50.pth > logs/bt.en.c100.r50.log &

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


############################################### Test commands ######################################################
####################################################################################################################

# DIR="saved_models"

# # Enable nullglob so the loop simply skips if no .pth files are found
# shopt -s nullglob

# for filepath in "$DIR"/*.pth; do
  
#   # Extract just the filename (e.g., "byol.c10.r18.pth")
#   filename=$(basename "$filepath")
  
#   # Use awk to split by '.' and print the 3rd last field (NF is Number of Fields)
#   ds_code=$(echo "$filename" | awk -F'.' '{print $(NF-2)}')
  
#   # Map the extracted code to the full dataset name
#   case "$ds_code" in
#     c10)  
#       dataset="cifar10" 
#       ;;
#     c100) 
#       dataset="cifar100" 
#       ;;
#     *)    
#       dataset="unknown_dataset ($ds_code)" 
#       ;;
#   esac
  
#   echo "Found model: $filepath"
#   echo "Dataset: $dataset"
#   python test.py --dataset "$dataset" --model resnet18 --saved_path "$filepath" --cmet --gpu 0

# done

# # Turn off nullglob
# shopt -u nullglob

# # 