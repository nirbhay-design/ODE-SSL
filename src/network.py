import math
import torch
import torchvision 
import torch.nn as nn 
import torch.nn.functional as F
from torchdiffeq import odeint
import warnings; warnings.filterwarnings("ignore")

class MLP(nn.Module): # MLP for linear protocol
    def __init__(self, in_features, num_classes, mlp_type="linear"):
        super().__init__()
        if mlp_type == "linear":
            print("===> using linear mlp")
            self.mlp = nn.Sequential(
                nn.Linear(in_features, num_classes)
            )
        else:
            print("===> using hiddin mlp")
            self.mlp = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.ReLU(),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        return self.mlp(x)

class CARL_mlp(nn.Module): # pred and proj net for carl
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

class Network(nn.Module):
    def __init__(self, model_name = 'resnet18', pretrained = False, proj_dim = 128, ode_steps = 10, algo_type="nodel", carl_hidden = 4096):
        super().__init__()
        if model_name == 'resnet50':
            model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif model_name  == 'resnet18':
            model = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        else:
            print(f"{model_name} model type not supported")
            model = None

        module_keys = list(model._modules.keys())
        self.feat_extractor = nn.Sequential()
        for key in module_keys[:-1]:
            if key == "maxpool": # don't add maxpool layer
                continue
            module_key = model._modules.get(key, nn.Identity())
            self.feat_extractor.add_module(key, module_key)

        if not pretrained:
            in_feat = self.feat_extractor.conv1.in_channels
            out_feat = self.feat_extractor.conv1.out_channels
            self.feat_extractor.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, bias=False)

        self.classifier_infeatures = model._modules.get(module_keys[-1], nn.Identity()).in_features

        self.algo_type = algo_type

        # so far general feature extractor ($h_{\theta}$)
        if self.algo_type in ["nodel", "carl"]:
            self.ode_steps = ode_steps
            self.ode_block = ODEBlock(ODENetwork(self.classifier_infeatures), steps=self.ode_steps)

            # This is projection head
            if algo_type == 'carl':
                self.proj = CARL_mlp(in_features = self.classifier_infeatures, hidden_dim = carl_hidden, out_features = proj_dim)
            else:
                self.proj = CARL_mlp(self.classifier_infeatures, 2*self.classifier_infeatures, proj_dim)
                # self.proj = nn.Linear(self.classifier_infeatures, proj_dim)
        elif self.algo_type in ["florel"]:
            self.proj = nn.Sequential(nn.Linear(self.classifier_infeatures, proj_dim),
                                      FloReLBlock(
                                          FloReLproj(proj_dim),
                                          steps = ode_steps
                                        )
                                    )
        elif self.algo_type in ["lema"]:
                self.proj = CARL_mlp(self.classifier_infeatures, 2*self.classifier_infeatures, proj_dim)

                                        
    def forward(self, x, t = None, test=None):
        features = self.feat_extractor(x).flatten(1)
        if test:
            return {"features": features}
        
        if self.algo_type in ["nodel", "carl"]:
            cont_dynamics = self.ode_block(features)
            if t is None:
                proj_features = self.proj(cont_dynamics[-1])
            else:
                proj_features = self.proj(cont_dynamics[t,torch.arange(x.shape[0],device=x.device)])
            return {"features": features, 
                    "cont_dyn": cont_dynamics, 
                    "proj_features": proj_features} # 2048/512, embedding dynamics
        
        elif self.algo_type in ["florel"]:
            proj = self.proj(features)
            return {"features": features,
                    "proj_features": proj["output"],
                    "logprob": proj["logprob"]}
        
        elif self.algo_type in ["lema"]:
            proj = self.proj(features)
            return {"features": features,
                    "proj_features": proj}
    


if __name__ == "__main__":
    device=torch.device('cuda:0')
    # network = Network(model_name = 'resnet50', pretrained=False, algo_type='florel', ode_steps=2, carl_hidden = 4096, proj_dim = 256)
    # # mlp = MLP(network.classifier_infeatures, num_classes=10, mlp_type='hidden')
    # network = network.to(device)
    # x = torch.rand(2,3,224,224,device=device)
    # # t = torch.randint(0, 10, size=(x.shape[0],))
    # # t = t.to(device)
    # output = network(x)
    # print(output["features"].shape)
    # # print(output["cont_dyn"].shape)
    # print(output["logprob"].shape)
    # print(output["proj_features"].shape)

    # contrastive loss on proj_feat, representations are feat, MLP on feat 
    energynet = EnergyNet(128)
    energynet = energynet.to(device)
    a = torch.rand(10,128, device=device)
    out = energynet.langevin_sampling(a, a)
    print(out.shape)
    print(out.isnan().sum())
    print(out.min(), out.max())
    