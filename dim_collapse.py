import torch
import torch.nn as nn
import torch.optim as optim
from src.network import BaseEncoder
from train_utils import load_dataset, progress, yaml_loader, get_features_labels, format_time
import argparse 
from functools import partial
import torch.nn.functional as F
import time, os
import matplotlib.pyplot as plt
import seaborn as sns
import copy 

def get_args():
    parser = argparse.ArgumentParser(description="Training script for linear probing")

    # basic experiment settings
    parser.add_argument("--dataset", type=str, default = "cifar10", required=True, help="dataset name")
    parser.add_argument("--saved_path", type=str, nargs='+', default = ["model.pth"], help="path for pretrained model")
    parser.add_argument("--gpu", type=int, default = 0, help="gpu_id")
    parser.add_argument("--model", type=str, default="resnet18", help="resnet18/resnet50/vit")
    parser.add_argument("--verbose", action="store_true", help="verbose or not")
    parser.add_argument("--nw", type=int, default = 4, help="num workers for dataloading")
    parser.add_argument("--pf", type=int, default = 4, help="prefetch factor for dataloading")
    parser.add_argument("--aug", type=str, default = "v1", help="augmentation strategy")
    parser.add_argument("--save_plot", type=str, default="DC.img100.pdf", help="file for saving the plot")
    parser.add_argument("--data_path", type=str, default=None, help="path to dataset")
    args = parser.parse_args()
    return args

def plot_embedding_singular_values(embeddings, save_path):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 6))

    colors = [
        "#800080",  # Purple
        '#0072B2',  # Deep Blue
        '#D55E00',  # Burnt Orange
        '#009E73',  # Bluish Green
        '#C0392B',  # Strong Red
        '#CC79A7',  # Dusky Pink
        '#E69F00',  # Goldenrod
        '#56B4E9',  # Sky Blue
        '#8C564B',  # Saddle Brown
        '#17BECF',  # Vibrant Teal
        '#34495E'   # Slate Charcoal
    ]

    for i, (algo, E) in enumerate(embeddings.items()):
        B, D = E.shape
        E = E - E.mean(axis = 0) # mean centered
        # 2. Compute the uncentered covariance matrix
        C = (E.T @ E) / (B - 1)

        # 3. Compute Singular Values
        # Because C is a symmetric positive semi-definite matrix, 
        # you can use svdvals. 
        S = torch.linalg.svdvals(torch.from_numpy(C))

        # 4. Plot the "Beautiful Curve"
        
        plt.plot(S.numpy(), linewidth=2.5, label = algo, color=colors[i % len(colors)])

    plt.yscale('log')
    plt.title('Singular Value Spectrum', fontsize=15, pad=15, fontweight='bold')
    plt.xlabel('Singular Value Index', fontsize=12)
    plt.ylabel('Singular Value Magnitude (Log Scale)', fontsize=12)
    plt.xlim(0, D)
    plt.legend(fontsize=10, loc='best', framealpha=0.7, shadow=False)
    # plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def load_model(model, saved_path, dataset, device): 
    encoder = BaseEncoder(model_name=model, pretrained=False, large = (dataset == "img100"))    
    print(encoder.load_state_dict(torch.load(saved_path, map_location=device)))
    encoder = encoder.to(device)
    return encoder 

if __name__ == "__main__":
    args = get_args()
    print(args)

    pt1 = time.perf_counter()

    config = yaml_loader("configs/test.yaml")
    config["dataset"][args.dataset]["params"]["num_workers"] = args.nw # set the number of workers for data loading 
    config["dataset"][args.dataset]["params"]["prefetch_factor"] = args.pf

    if args.data_path:
        config["dataset"][args.dataset]["params"]["data_dir"] = args.data_path

    train_dl, train_dl_mlp, test_dl, train_dataset, test_dataset = load_dataset(
        dataset_name = args.dataset,
        distributed = False,
        aug = args.aug,
        **config["dataset"][args.dataset]["params"])
    
    device = torch.device(f"cuda:{args.gpu}")

    all_embeddings = {}

    for path in args.saved_path: 
        encoder = load_model(args.model, path, args.dataset, device)
        emb_key = '.'.join(filter(lambda x: x!= "dist", path.split("/")[-1].split('.')[:-1]))  
        print(emb_key)    
        embeddings = get_features_labels(encoder, train_dl_mlp, device, return_logs = args.verbose)
        # print(embeddings["features"].shape)
        all_embeddings[emb_key] = embeddings["features"]
  
    plot_embedding_singular_values(all_embeddings, os.path.join("plots", args.save_plot))

    pt2 = time.perf_counter()
    print(f"time: {format_time(pt2 - pt1)}")