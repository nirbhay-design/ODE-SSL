import math
import torch
import torchvision 
import torch.nn as nn 
import torch.nn.functional as F
from torchdiffeq import odeint

class ODENetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = 2*input_dim
        self.odenet = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_dim, input_dim))
        )

    def forward(self, t, x):
        t = torch.tensor(t).repeat((x.shape[0],1))
        return self.odenet(torch.cat([x,t], dim = -1))
    
class ODEBlock(nn.Module):
    def __init__(self, odefun, T=1.0, steps=10, method='rk4'):
        super().__init__()
        self.odefun = odefun
        self.t_grid = torch.linspace(0.0, T, steps)
        self.method = method

    def forward(self, x):
        self.t_grid = self.t_grid.to(x.device)
        output = odeint(self.odefun, x, self.t_grid, method=self.method)
        return output

class FloReLproj(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = 2 * input_dim
        self.odenet = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(input_dim + 1, hidden_dim)),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_dim, input_dim))
        )
    
    def forward(self, t, x):
        t = torch.tensor(t, device=x.device, dtype=x.dtype).repeat((x.shape[0],1))
        return self.odenet(torch.cat([x,t], dim = -1))

class FloReLBlock(nn.Module):
    def __init__(self, odefun, T=1.0, steps=10, trace_steps=2, method='rk4'):
        super().__init__()
        self.odefun = odefun
        self.t_grid = torch.linspace(0, T, steps)
        self.method = method
        self.trace_steps = trace_steps

    def divergence_estimate(self, t, z, e = None):
        if e is None:
            e = torch.randint(0, 2, z.shape, device=z.device, dtype=z.dtype) * 2.0 - 1.0 # -1/1 rademacher
        
        z = z.clone().detach().requires_grad_(True)
        f = self.odefun(t, z)
        etf = torch.autograd.grad((f * e).sum(), z, create_graph=True)[0]
        div_est = (etf*e).sum(dim = -1)  

        # z.requires_grad_(False)
        z = z.detach()
        return f, div_est

    def augmented_dynamics(self, t, state): # state is [z, logp]
        z = state[:,:-1]
        # logp = state[:, -1:]
        
        f_z = None 
        div_est = torch.zeros(z.shape[0], device=z.device)
        for _ in range(self.trace_steps):
            f_z, div = self.divergence_estimate(t, z)
            div_est += div
        div_est /= self.trace_steps
        div_est = torch.clamp(div_est, -100.0, 100.0)

        augmented_state = torch.cat([f_z, -div_est.unsqueeze(1)], dim = -1)
        return augmented_state 

    def forward(self, x):
        self.t_grid = self.t_grid.to(x.device)
        initial_state = torch.cat([x, torch.zeros(x.shape[0], 1, device=x.device)], dim = -1)
        output = odeint(self.augmented_dynamics, initial_state, self.t_grid, method=self.method, rtol = 1e-6, atol = 1e-6)
        final_output = output[-1]
        rep, probsolve = final_output[:,:-1], final_output[:, -1]

        # base_log_prob = torch.distributions.MultivariateNormal(
        #     torch.zeros(x.shape[1], device=x.device),
        #     torch.eye(x.shape[1], device=x.device)
        # ).log_prob(rep)
        # probsolve is -\int_{0}^{1} tr(f')

        base_log_prob = -0.5 * rep.norm(dim = -1).pow(2) # sufficient for training 
        return {"output": rep, "logprob": base_log_prob + probsolve}

class EnergyNet(nn.Module):
    def __init__(self, z_dim, eta = 1e-4, steps = 30):
        super().__init__()
        hidden = z_dim * 2
        self.enet = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(2 * z_dim, hidden)),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1),
            nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1),
            nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, 1)),
        )

        # parameters for langevin sampling 
        self.eta = eta
        self.steps = steps 

    def forward(self, z1, z2):
        # returns energy between z1 and z2
        z = torch.cat([z1,z2], dim = -1)
        return self.enet(z)

    def langevin_sampling(self, z_cond, z_0 = None):
        self.eval()
        z_cond = z_cond.clone().detach()
        if z_0 is None:
            z = torch.randn_like(z_cond)
        else:
            z = z_0.clone().detach()
        z.requires_grad_(True)
        for _ in range(self.steps):
            e = self(z, z_cond).squeeze().sum()
            grad = torch.autograd.grad(e, z, create_graph=False)[0]
            z = z - self.eta * torch.clamp(grad, -self.eta, self.eta) + math.sqrt(2 * self.eta) * torch.randn_like(z_cond) # langevin dynamics
            z = z.detach()
            z.requires_grad_(True)
        self.train()
        return z.detach()

def train_lema( # low energy manifolds based representation learning 
        model, mlp, energy_model, train_loader, train_loader_mlp,
        test_loader, lossfunction, lossfunction_mlp, 
        optimizer, mlp_optimizer, energy_optimizer, opt_lr_schedular, 
        eval_every, n_epochs, n_epochs_mlp, device_id, eval_id, tsne_name, return_logs=False): 
    

    print(f"### LEMa Training begins")
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    energy_model = energy_model.to(device)
    for epochs in range(n_epochs):
        model.train()
        energy_model.train()
        cur_loss = 0
        en_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, _) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)


            output = model(data)
            output_cap = model(data_cap)

            proj_feat = output["proj_features"]
            proj_feat_cap = output_cap["proj_features"]

            feat = output["features"]
            feat_cap = output_cap["features"] 

            esample = energy_model.langevin_sampling(feat, z_0 = feat)
            esample_cap = energy_model.langevin_sampling(feat_cap, z_0 = feat_cap)
            
            loss_con = lossfunction(proj_feat, proj_feat_cap) + F.mse_loss(feat, esample.detach()) + F.mse_loss(feat_cap, esample_cap.detach())
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            # training energy model
            # pos_energy = energy_model(feat.detach(), feat_cap.detach())
            # neg_energy = energy_model(esample.detach(), esample_cap.detach())

            pos_energy = energy_model(feat_cap.detach(), feat.detach()) + energy_model(feat.detach(), feat_cap.detach())
            neg_energy = energy_model(esample_cap.detach(), feat.detach()) + energy_model(esample.detach(), feat_cap.detach())
            energy_loss = pos_energy.mean() - neg_energy.mean()

            energy_optimizer.zero_grad()
            energy_loss.backward()
            energy_optimizer.step()
            
            cur_loss += loss_con.item() / (len_train)
            en_loss += energy_loss.item() / len_train
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), en_loss = energy_loss.item(), GPU = device_id)
        
        opt_lr_schedular.step()
            
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f} energy_loss: {en_loss:.3f}")

    print("### TSNE starts")
    make_tsne_for_dataset(model, test_loader, device_id, "lema", return_logs = return_logs, tsne_name = tsne_name)

    print("### MLP training begins")
    train_mlp(
        model, mlp, train_loader_mlp, test_loader, 
        lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
        device_id, eval_id, return_logs = return_logs)

    return model