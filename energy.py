"""
latent_ebm_contrastive.py

A self-contained PyTorch example implementing the idea you described:
- Encoder -> latent z
- Projector -> projection for contrastive (InfoNCE)
- Energy network defined on latent z
- Langevin sampling in latent space starting from Gaussian noise to get z_sample
- Decoder (optional) to map z_sample -> x_hat
- Outer loss: reconstruction (or perceptual) + contrastive

Dataset: CIFAR-10 (default). Easily swappable.

Notes:
- This implementation *detaches* the Langevin sampling (no backprop through inner loop) for stability.
- Uses latent-space EBM (cheaper & faster). If you want pixel-space EBM, swap encoder/decoder usage accordingly.

Run:
    python latent_ebm_contrastive.py

Requires: torch, torchvision

"""

import argparse
import math
import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------- Utilities ---------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------- Models ------------------------------------
# Simple encoder for CIFAR-like images. Replace with ResNet for better results.
class SimpleEncoder(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.fc = nn.Linear(256, z_dim)

    def forward(self, x):
        f = self.conv(x).view(x.size(0), -1)
        z = self.fc(f)
        return z


class Projector(nn.Module):
    def __init__(self, z_dim=128, p_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim, p_dim)
        )

    def forward(self, z):
        return self.net(z)


class Decoder(nn.Module):
    """Simple decoder to reconstruct CIFAR images from latent z.
    Not designed for SOTA image quality, but adequate for experiments.
    """
    def __init__(self, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3 * 32 * 32),
        )

    def forward(self, z):
        x = self.net(z)
        x = x.view(z.size(0), 3, 32, 32)
        return x


class EnergyNet(nn.Module):
    """Energy network defined on latent z. Outputs scalar energy per sample.

    Note: spectral normalization or gradient penalties can be added for stability.
    """
    def __init__(self, z_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        # returns shape (batch,)
        e = self.net(z).squeeze(-1)
        return e


# --------------------------- Losses ------------------------------------

def info_nce_loss(p1: torch.Tensor, p2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Simple InfoNCE loss with in-batch negatives. Assumes p1 and p2 are L2-normalized or we'll normalize here."""
    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=1)
    batch_size = p1.size(0)

    # similarities: (2B x 2B) matrix if we concatenate views; we'll use simpler in-batch formulation
    positives = torch.sum(p1 * p2, dim=1)  # (B,)

    # denominator uses all negatives from the batch (including positives in numerator)
    logits = torch.mm(p1, p2.t()) / temperature  # (B, B)

    # for each i, positive is at position i
    labels = torch.arange(batch_size, device=p1.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_i + loss_j)
    return loss


# --------------------------- Sampler ----------------------------------

def langevin_latent_sampling(
    z_init: torch.Tensor,
    energy_net: nn.Module,
    steps: int = 30,
    step_size: float = 1e-3,
    noise_scale: float = 1e-2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Run Langevin updates in latent space to find low-energy z samples.

    This implementation *detaches* inside the loop to avoid building a huge graph.
    Returns final z_sample detached from the sampling history.
    """
    z = z_init.clone().detach().to(device)
    z = z.requires_grad_(True)

    for t in range(steps):
        # compute scalar energy for the batch and take gradients
        e = energy_net(z).sum()
        grad = torch.autograd.grad(e, z, create_graph=False)[0]

        # gradient descent + gaussian noise (Langevin)
        z = z - 0.5 * step_size * grad
        z = z + torch.randn_like(z) * (noise_scale * math.sqrt(step_size))

        # detach to keep memory low (we won't backprop through inner steps)
        z = z.detach()
        z = z.requires_grad_(True)

    return z.detach()


# --------------------------- Training ---------------------------------

def train(
    args,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Data
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    train_ds = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Models
    encoder = SimpleEncoder(z_dim=args.z_dim).to(device)
    projector = Projector(z_dim=args.z_dim, p_dim=args.p_dim).to(device)
    energy_net = EnergyNet(z_dim=args.z_dim).to(device)
    decoder = Decoder(z_dim=args.z_dim).to(device) if args.use_decoder else None

    params = list(encoder.parameters()) + list(projector.parameters())
    if decoder is not None:
        params += list(decoder.parameters())
    # energy updated separately with its own optimizer (optionally)
    opt_main = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    opt_energy = torch.optim.AdamW(energy_net.parameters(), lr=args.lr_energy, weight_decay=1e-6)

    # training loop
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        projector.train()
        energy_net.train()
        if decoder is not None:
            decoder.train()

        running_loss = 0.0
        running_contrastive = 0.0
        running_recon = 0.0
        running_energy_mean = 0.0

        for it, (x, _) in enumerate(train_loader):
            x = x.to(device)

            # two augmentations (x1, x2)
            x1 = x  # already augmented by transform
            # create a second augmentation on-the-fly
            x2 = transform(x.cpu()).to(device)

            # encode & project
            z1 = encoder(x1)
            p1 = projector(z1)

            z2 = encoder(x2)
            p2 = projector(z2)

            # Langevin sampling in latent space
            z0 = torch.randn_like(z1)
            z_sample = langevin_latent_sampling(
                z0,
                energy_net,
                steps=args.langevin_steps,
                step_size=args.langevin_step_size,
                noise_scale=args.langevin_noise,
                device=device,
            )

            # decode sampled latent to image (optional)
            if decoder is not None:
                x_sample = decoder(z_sample)
                recon_loss = F.mse_loss(x_sample, x2)
            else:
                # if no decoder, do a latent-space reconstruction: ||z2 - z_sample||^2
                recon_loss = F.mse_loss(z_sample, z2)

            contrastive = info_nce_loss(p1, p2, temperature=args.temperature)

            loss = args.lambda_recon * recon_loss + args.lambda_contrast * contrastive

            # update main networks (encoder/projector/decoder)
            opt_main.zero_grad()
            loss.backward()
            opt_main.step()

            # update energy network separately: train to assign low energy to z_sample and higher energy to z0 (or z2)
            # simple objective: E(z_sample) + max(0, margin - E(z_pos)) where z_pos = z2
            energy_pos = energy_net(z2)
            energy_sample = energy_net(z_sample.detach())
            # we want E(pos) to be low, E(sample) to be low too, but push random init to higher energy
            # a typical EBM loss is: E(pos) - E(sample) (minimize)
            energy_loss = (energy_pos - energy_sample).mean()

            opt_energy.zero_grad()
            energy_loss.backward()
            opt_energy.step()

            running_loss += loss.item()
            running_contrastive += contrastive.item()
            running_recon += recon_loss.item()
            running_energy_mean += energy_pos.mean().item()

            if (it + 1) % args.print_every == 0:
                n = args.print_every
                print(
                    f"Epoch [{epoch}/{args.epochs}] Iter [{it+1}/{len(train_loader)}] ",
                    f"loss={running_loss / n:.4f}",
                    f"contrast={running_contrastive / n:.4f}",
                    f"recon={running_recon / n:.4f}",
                    f"E_pos={running_energy_mean / n:.4f}",
                )
                running_loss = running_contrastive = running_recon = running_energy_mean = 0.0

        # end epoch: optional checkpoint
        if epoch % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'encoder': encoder.state_dict(),
                'projector': projector.state_dict(),
                'energy': energy_net.state_dict(),
                'decoder': decoder.state_dict() if decoder is not None else None,
            }, os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pth"))


# --------------------------- Argparse ---------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_energy', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--p_dim', type=int, default=64)

    parser.add_argument('--langevin_steps', type=int, default=20)
    parser.add_argument('--langevin_step_size', type=float, default=1e-3)
    parser.add_argument('--langevin_noise', type=float, default=1e-2)

    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_contrast', type=float, default=1.0)

    parser.add_argument('--use_decoder', action='store_true')

    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--print_every', type=int, default=50)

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train(args)
