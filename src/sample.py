# pip install torch torchvision torchdiffeq
import math, time, os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchdiffeq import odeint  # direct backprop (stable)
from contextlib import nullcontext

# ========== Utils ==========
def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TwoCropsTransform:
    def __init__(self, base): self.base = base
    def __call__(self, x): return self.base(x), self.base(x)

def simclr_transforms(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1*size)//2*2+1, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

# ========== Model pieces ==========

def make_backbone():
    m = models.resnet50(weights=None)
    m.fc = nn.Identity()
    feat_dim = 2048
    return m, feat_dim

class ODEFunc(nn.Module):
    """
    Vector field g(t, z). Time-conditioning via scalar t concatenation.
    Uses GroupNorm + SiLU and spectral norm on the last layer for Lipschitz control.
    """
    def __init__(self, dim, hidden=1024, t_cond=True):
        super().__init__()
        self.t_cond = t_cond
        in_dim = dim + (1 if t_cond else 0)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.gn1 = nn.GroupNorm(32, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.gn2 = nn.GroupNorm(32, hidden)
        self.fc3 = nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, dim))

        # optional: scale final output to avoid large drifts at init
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, z):
        if self.t_cond:
            tfeat = torch.ones_like(z[:, :1]) * t
            z = torch.cat([z, tfeat], dim=1)
        h = self.fc1(z); h = self.gn1(h); h = F.silu(h)
        h = self.fc2(h); h = self.gn2(h); h = F.silu(h)
        dz = self.fc3(h) * self.scale
        return dz

class ODEBlock(nn.Module):
    """
    Fixed-step RK4 via torchdiffeq (method='rk4') for determinism & stability.
    Returns full trajectory if steps>2 (for regularizers).
    """
    def __init__(self, func, T=1.0, steps=8, rtol=1e-4, atol=1e-4, method='rk4'):
        super().__init__()
        self.func, self.T, self.steps = func, float(T), int(steps)
        self.rtol, self.atol, self.method = rtol, atol, method
        assert self.steps >= 2, "Use at least 2 steps."
        self.register_buffer('t_grid', torch.linspace(0.0, self.T, self.steps))

    def forward(self, z0, return_traj=True):
        t = self.t_grid.to(z0)
        z_traj = odeint(self.func, z0, t, rtol=self.rtol, atol=self.atol, method=self.method,
                        options={'step_size': self.T / (self.steps - 1)} if self.method == 'rk4' else None)
        if return_traj:  # [S,B,D]
            return z_traj
        else:
            return z_traj[-1]

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )
    def forward(self, z): return self.net(z)

class NeuralODESimCLR(nn.Module):
    def __init__(self, proj_dim=128, ode_T=1.0, ode_steps=8, hidden=1024):
        super().__init__()
        self.backbone, enc_dim = make_backbone()
        self.ode = ODEBlock(ODEFunc(enc_dim, hidden=hidden, t_cond=True),
                            T=ode_T, steps=ode_steps, method='rk4')
        self.proj = ProjectionHead(enc_dim, proj_dim)

    def forward_once(self, x):
        z0 = self.backbone(x)  # (B,D)
        traj = self.ode(z0, return_traj=True)  # (S,B,D)
        zT = traj[-1]
        p = F.normalize(self.proj(zT), dim=1)
        return p, traj

    def forward(self, x1, x2):
        p1, traj1 = self.forward_once(x1)
        p2, traj2 = self.forward_once(x2)
        return (p1, traj1), (p2, traj2)

# ========== Losses ==========

def nt_xent(z1, z2, tau=0.2):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                     # (2B,D), already L2-normalized
    logits = (z @ z.t()) / tau
    mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, -float('inf'))
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)
    return F.cross_entropy(logits, labels)

def path_smoothness(traj):
    # traj: (S,B,D)
    diffs = traj[1:] - traj[:-1]
    return (diffs.pow(2).sum(dim=2).mean())

def energy_from_field(func, t_grid, traj):
    # Approximate \int ||g(t,z)||^2 dt with trapezoidal rule on the same grid
    # traj: (S,B,D); t_grid: (S,)
    S = traj.size(0)
    B = traj.size(1)
    E = 0.0
    for k in range(S):
        t = t_grid[k]
        z = traj[k].detach()  # avoid second-order grads; stable
        g = func(t, z)
        E = E + (g.pow(2).sum(dim=1).mean())
    E = E / S
    return E

def traj_alignment(traj1, traj2):
    # Sum over time of ||z1(t) - z2(t)||^2
    return (traj1 - traj2).pow(2).sum(dim=2).mean()

# ========== EMA (optional but helps) ==========
class EMA:
    def __init__(self, model, decay=0.996):
        self.shadow = {}
        self.decay = decay
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)

    def apply_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

# ========== Training loop ==========

def main():
    set_seed(123)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Data
    transform = TwoCropsTransform(simclr_transforms(224))
    train_set = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # Model
    model = NeuralODESimCLR(proj_dim=128, ode_T=1.0, ode_steps=8, hidden=1024).to(device)

    # Freeze encoder BN stats during pretrain (robust for SimCLR)
    model.backbone.train()
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.0
            m.track_running_stats = True

    # Optimizer & schedule
    base_lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=400)  # adjust T_max to your steps/epochs
    ema = EMA(model, decay=0.996)

    # Loss weights
    tau = 0.2
    lam_path = 1e-3       # path smoothness
    lam_energy = 1e-3     # energy (vector-field L2)
    lam_align = 5e-4      # trajectory alignment between two views

    model.train()
    for step, ((x1, x2), _) in enumerate(loader, start=1):
        x1, x2 = x1.to(device), x2.to(device)
        optimizer.zero_grad(set_to_none=True)

        ctx = torch.cuda.amp.autocast(enabled=amp)
        with ctx:
            (p1, traj1), (p2, traj2) = model(x1, x2)
            loss_ntx = nt_xent(p1, p2, tau=tau)
            loss_path = path_smoothness(traj1) + path_smoothness(traj2)
            loss_align = traj_alignment(traj1, traj2)

            # Energy from vector field (same grid the ODE used)
            t_grid = model.ode.t_grid.to(device)
            loss_energy = energy_from_field(model.ode.func, t_grid, traj1) + \
                          energy_from_field(model.ode.func, t_grid, traj2)

            loss = loss_ntx + lam_path*loss_path + lam_energy*loss_energy + lam_align*loss_align

        # Backprop
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        if step % 50 == 0:
            print(f"[{step}] loss={loss.item():.4f} | ntx={loss_ntx.item():.4f} | "
                  f"path={loss_path.item():.4f} | energy={loss_energy.item():.4f} | align={loss_align.item():.4f}")

    # (Optional) swap to EMA weights before eval or saving
    ema.apply_to(model)
    torch.save(model.state_dict(), "neuralode_simclr.pth")
    print("Saved to neuralode_simclr.pth")

if __name__ == "__main__":
    main()
