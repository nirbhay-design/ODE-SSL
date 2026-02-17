import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

class VICRegLoss(nn.Module):
    def __init__(self, _lambda = 25.0, mu = 25.0, nu = 1.0):
        super().__init__()
        self._lambda = _lambda
        self.mu = mu    
        self.nu = nu

    def off_diagonal(self, x):
        n, _ = x.shape 
        return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

    def forward(self, za, zb):
        sim_loss = F.mse_loss(za, zb)

        za_var = torch.sqrt(za.var(dim = 0) + 1e-04)
        zb_var = torch.sqrt(zb.var(dim = 0) + 1e-04)
        var_loss = 0.5 * (torch.mean(F.relu(1-za_var)) + torch.mean(F.relu(1-zb_var)))

        za = za - za.mean(dim = 0)
        zb = zb - zb.mean(dim = 0)

        N, D = za.shape 
        cov_za = (za.T @ za) / (N - 1)
        cov_zb = (zb.T @ zb) / (N - 1)

        cov_loss = self.off_diagonal(cov_za).pow(2).sum().div(D) +\
                    self.off_diagonal(cov_zb).pow(2).sum().div(D)

        return self._lambda * sim_loss + self.mu * var_loss + self.nu * cov_loss  

def train_vicreg_sc(
        model, mlp, energy_model, train_loader, train_loader_mlp,
        test_loader, lossfunction, lossfunction_mlp, energy_optimizer,
        optimizer, mlp_optimizer, opt_lr_schedular, 
        eval_every, n_epochs, n_epochs_mlp, device_id, eval_id, tsne_name, return_logs=False): 
    
    print(f"### VICReg-SC Training begins")

    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    energy_model = energy_model.to(device)
    for epochs in range(n_epochs):
        model.train()
        energy_model.train()
        en_loss = 0
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, _) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)
            
            output = model(data)
            output_cap = model(data_cap)

            feat, proj_feat = output["features"], output["proj_features"]
            feat_cap, proj_feat_cap = output_cap["features"], output_cap["proj_features"]

            esample = energy_model.langevin_sampling(feat)
            esample_cap = energy_model.langevin_sampling(feat_cap)

            loss_con = lossfunction(proj_feat, proj_feat_cap) + 0.1 * \
                (F.mse_loss(feat, esample.detach()) + F.mse_loss(feat_cap, esample_cap.detach())) 
            
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

    print("### TSNE starts")
    make_tsne_for_dataset(model, test_loader, device_id, 'vicreg-sc', return_logs = return_logs, tsne_name = tsne_name)

    print("### MLP training begins")

    train_mlp(
        model, mlp, train_loader_mlp, test_loader, 
        lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
        device_id, eval_id, return_logs = return_logs)

    return model


def train_vicreg(
        model, mlp, train_loader, train_loader_mlp,
        test_loader, lossfunction, lossfunction_mlp, 
        optimizer, mlp_optimizer, opt_lr_schedular, 
        eval_every, n_epochs, n_epochs_mlp, device_id, eval_id, tsne_name, return_logs=False): 
    
    print(f"### VICReg Training begins")

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

            feats, proj_feat = output["features"], output["proj_features"]
            feats_cap, proj_feat_cap = output_cap["features"], output_cap["proj_features"]

            loss_con = lossfunction(proj_feat, proj_feat_cap)
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
              
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    print("### TSNE starts")
    make_tsne_for_dataset(model, test_loader, device_id, 'vicreg', return_logs = return_logs, tsne_name = tsne_name)

    print("### MLP training begins")

    train_mlp(
        model, mlp, train_loader_mlp, test_loader, 
        lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
        device_id, eval_id, return_logs = return_logs)

    return model