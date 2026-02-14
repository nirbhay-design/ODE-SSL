import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

class BarlowTwinLoss(nn.Module):
    def __init__(self, lambd = 0.1):
        super().__init__()
        self.lambd = lambd

    def forward(self, za, zb):
        N, D = za.shape

        za = (za - za.mean(0, keepdim=True)) / za.std(0, keepdim=True)
        zb = (zb - zb.mean(0, keepdim=True)) / zb.std(0, keepdim=True)

        C = torch.mm(za.T, zb) / N # DxD

        I = torch.eye(D, device=za.device)

        diff = (I - C).pow(2)

        diag_elem = torch.diag(diff)

        diff.fill_diagonal_(0.0)

        return diag_elem.sum() + self.lambd * diff.sum()