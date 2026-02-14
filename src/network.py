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
    
# Energy network regularization
class EnergyScoreNet(nn.Module):
    def __init__(self, z_dim, eta = 1e-4, steps = 30, sigma = 1e-3, delta = 0.1, net_type = "score"):
        """
        net_type: score / energy
        """
        super().__init__()
        hidden = z_dim * 2
        self.snet = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(z_dim, hidden)),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1),
            nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1)
        )

        self.net_type = net_type 
        self.sigma = sigma 
        self.delta = delta 

        if self.net_type == "score":
            self.snet.append(nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, z_dim)))
        else:
            self.snet.append(nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, 1)))

        # parameters for langevin sampling 
        self.eta = eta
        self.steps = steps 

    def forward(self, z):
        return self.snet(z)

    def langevin_sampling(self, z = None):
        self.eval()
        if z is None:
            z = torch.randn_like(z)
        else:
            z = z.clone().detach()
        z.requires_grad_(True)
        for _ in range(self.steps):
            if self.net_type == "score":
                grad = self(z)
            else:
                e = self(z).squeeze().sum()
                grad = -torch.autograd.grad(e, z, create_graph=False)[0] # -\nabla_{z} E(z)
            z = z + self.eta * torch.clamp(grad, -self.delta, self.delta) + math.sqrt(2 * self.eta) * torch.randn_like(z) # langevin dynamics
            z = z.detach()
            z.requires_grad_(True)
        self.train()
        return z.detach()
    
    def dsm_loss(self, z):
        epsilon = torch.randn_like(z)
        z_hat = z + self.sigma * epsilon
        if self.net_type == "score":
            s = self(z_hat)
            loss = 0.5 * (self.sigma * s + epsilon).pow(2).sum(dim = -1).mean()
        elif self.net_type == "energy":
            z_hat.requires_grad_(True)
            e = self(z_hat).sum()
            s = torch.autograd.grad(e, z_hat, create_graph=True)[0]
            loss = 0.5 * (self.sigma * s - epsilon).pow(2).sum(dim = -1).mean()
        return loss 

# proj_dim = 128, ode_steps = 10, algo_type="nodel", carl_hidden = 4096, byol_hidden=4096, pred_dim = 512, barlow_hidden = 8192, vae_out = 256
class Network(nn.Module):
    def __init__(self, model_name = 'resnet18', pretrained = False, **kwargs):
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
            self.feat_extractor.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False)

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

        elif self.algo_type in ["dailema"]:
            self.proj = VAE_linear(self.classifier_infeatures, vae_out)

        elif self.algo_type in ["scalre"]: # Score Alignment for representation learning 
            self.proj = CARL_mlp(self.classifier_infeatures, 2*self.classifier_infeatures, proj_dim)

        elif self.algo_type in ["byol-sc"]:
            self.proj = CARL_mlp(in_features = self.classifier_infeatures, hidden_dim = byol_hidden, out_features = proj_dim)

        elif self.algo_type in ["simsiam-sc"]:
            prev_dim = self.classifier_infeatures
            self.proj = nn.Sequential(
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(),
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim)
            )
            self.pred = nn.Sequential(
                nn.Linear(prev_dim, pred_dim, bias=False),
                nn.BatchNorm1d(pred_dim),
                nn.ReLU(),
                nn.Linear(pred_dim, prev_dim)
            )           
        
        elif self.algo in ["bt-sc", "vicreg-sc"]:
            self.proj = nn.Sequential(
                nn.Linear(self.classifier_infeatures, barlow_hidden, bias=False),
                nn.BatchNorm1d(barlow_hidden, bias=False),
                nn.ReLU(),
                nn.Linear(barlow_hidden, barlow_hidden, bias=False),
                nn.BatchNorm1d(barlow_hidden),
                nn.ReLU(),
                nn.Linear(barlow_hidden, proj_dim)
            )

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
        
        elif self.algo_type in ["lema", "scalre", "bt-sc", "byol-sc", "vicreg-sc"]:
            proj = self.proj(features)
            return {"features": features,
                    "proj_features": proj}
        
        elif self.algo_type in ["simsiam-sc"]:
            proj = self.proj(features)
            pred = self.pred(proj)
            return {"features": features,
                    "proj_features": proj,
                    "pred_features": pred}
        
        elif self.algo_type in ["dailema"]:
            proj = self.proj(features)
            return {"features": features,
                    "mu": proj["mu"],
                    "log_var": proj["log_var"]}


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
    # energynet = EnergyNet(128)
    # energynet = energynet.to(device)
    

    energy_score_net = EnergyScoreNet(z_dim = 128, net_type = "score").to(device)

    a = torch.rand(10,128, device=device)
    out = energy_score_net.langevin_sampling(a)
    print(out.shape)
    print(out.isnan().sum())
    print(out.min(), out.max())