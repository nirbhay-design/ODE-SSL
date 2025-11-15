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

class FloReLBlock(nn.Module):
    def __init__(self, odefun, T=1.0, steps=10, trace_steps=10, method='rk4'):
        super().__init__()
        self.odefun = odefun
        self.t_grid = torch.tensor([0.0, T])
        self.method = method
        self.trace_steps = 10

    def divergence_estimate(self, t, z, e = None):
        if e is None:
            e = torch.randint(0, 2, z.shape, device=z.device, dtype=z.dtype) * 2.0 - 1.0 # -1/1 rademacher
        
        z.requires_grad_(True)
        f = self.odefun(t, z)
        etf = torch.autograd.grad((f * e).sum(), z, create_graph=True)[0]
        div_est = (etf*e).sum(dim = -1)  

        z.requires_grad_(False)
        return f, div_est

    def augmented_dynamics(self, t, state): # state is [z, logp]
        z = state[:,:-1]
        logp = state[:, -1:]
        
        f_z = None 
        div_est = torch.zeros(z.shape[0], device=z.device)
        for _ in range(self.trace_steps):
            f_z, div = self.divergence_estimate(t, z)
            div_est += div
        div_est /= self.trace_steps

        augmented_state = torch.cat([f_z, -div_est.unsqueeze(1)], dim = -1)
        return augmented_state 

    def forward(self, x):
        self.t_grid = self.t_grid.to(x.device)
        inital_state = torch.cat([x, torch.zeros(x.shape[0], 1, device=x.device)], dim = -1)
        output = odeint(self.augmented_dynamics, initial_state, self.t_grid, method=self.method)
        final_output = output[-1]
        
        return output

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

        # so far general feature extractor ($h_{\theta}$)
        self.ode_steps = ode_steps
        self.ode_block = ODEBlock(ODENetwork(self.classifier_infeatures), steps=self.ode_steps)

        # This is projection head
        if algo_type == 'carl':
            self.proj = CARL_mlp(in_features = self.classifier_infeatures, hidden_dim = carl_hidden, out_features = proj_dim)
        else:
            self.proj = CARL_mlp(self.classifier_infeatures, 2*self.classifier_infeatures, proj_dim)
            # self.proj = nn.Linear(self.classifier_infeatures, proj_dim)
            
        self.algo_type = algo_type

    def forward(self, x, t = None):
        features = self.feat_extractor(x).flatten(1)
        cont_dynamics = self.ode_block(features)
        if t is None:
            proj_features = self.proj(cont_dynamics[-1])
        else:
            proj_features = self.proj(cont_dynamics[t,torch.arange(x.shape[0],device=x.device)])
        return {"features": features, 
                "cont_dyn": cont_dynamics, 
                "proj_features": proj_features} # 2048/512, embedding dynamics
    


if __name__ == "__main__":
    device=torch.device('cuda:0')
    network = Network(model_name = 'resnet50', pretrained=False, algo_type='byol', ode_steps=10, carl_hidden = 4096, proj_dim = 256)
    # mlp = MLP(network.classifier_infeatures, num_classes=10, mlp_type='hidden')
    network = network.to(device)
    x = torch.rand(2,3,224,224,device=device)
    t = torch.randint(0, 10, size=(x.shape[0],))
    t = t.to(device)
    output = network(x,t)
    print(output["features"].shape)
    print(output["cont_dyn"].shape)
    print(output["proj_features"].shape)

    # contrastive loss on proj_feat, representations are feat, MLP on feat 

    