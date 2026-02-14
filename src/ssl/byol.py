import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

class BYOLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, online, target):
        online = F.normalize(online, dim = -1)
        target = F.normalize(target, dim = -1)

        return -2 * (online * target).sum(dim = -1).mean()

class BYOL_mlp(nn.Module): # pred and proj net for carl
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.mlp(x)