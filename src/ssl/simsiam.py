import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

class SimSiamLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, z):
        p = F.normalize(p, dim = -1)
        z = F.normalize(z, dim = -1)

        return -(p * z).sum(dim = -1).mean()

class simsiam_proj(nn.Module):
    def __init__(self, prev_dim):
        super().__init__()
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

    def forward(self, x):
        return self.proj(x) 

class simsiam_pred(nn.Module):
    def __init__(self, prev_dim, pred_dim):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Linear(prev_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, prev_dim)
        )
    def forward(self, x):
        return self.pred(x)

def train_simsiam_sc(
        model, energy_model, train_loader,
        lossfunction, optimizer, energy_optimizer, opt_lr_schedular, 
        n_epochs, device_id, eval_id, return_logs=False, progress=None): 
    

    print(f"### simsiam with SC-net Training begins")
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

            feat, proj_feat, pred_feat = output["features"], output["proj_features"], output["pred_features"]
            feat_cap, proj_feat_cap, pred_feat_cap = output_cap["features"], output_cap["proj_features"], output_cap["pred_features"]

            esample = energy_model.langevin_sampling(feat)
            esample_cap = energy_model.langevin_sampling(feat_cap)
            
            loss_con = 0.5 * (lossfunction(pred_feat, proj_feat_cap.detach()) + lossfunction(pred_feat_cap, proj_feat.detach())) + F.mse_loss(feat, esample.detach()) + F.mse_loss(feat_cap, esample_cap.detach())
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            energy_loss = 0.5 * (energy_model.dsm_loss(feat.detach()) + energy_model.dsm_loss(feat_cap.detach()))

            energy_optimizer.zero_grad()
            energy_loss.backward()
            energy_optimizer.step()
            
            cur_loss += loss_con.item() / (len_train)
            en_loss += energy_loss.item() / len_train
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), en_loss = energy_loss.item(), GPU = device_id)
        
        opt_lr_schedular.step()
            
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f} energy_loss: {en_loss:.3f}")

    return model

def train_simsiam(
        model, train_loader, lossfunction,
        optimizer, opt_lr_schedular, n_epochs, 
        device_id, eval_id, return_logs=False, progress=None): 
    

    print(f"### simsiam Training begins")
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, target) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)

            output = model(data)
            output_cap = model(data_cap)

            feats, proj_feat, pred_feat = output["features"], output["proj_features"], output["pred_features"]
            feats_cap, proj_feat_cap, pred_feat_cap = output_cap["features"], output_cap["proj_features"], output_cap["pred_features"]

            
            loss_con = 0.5 * (lossfunction(pred_feat, proj_feat_cap.detach()) + lossfunction(pred_feat_cap, proj_feat.detach()))
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
            
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    return model