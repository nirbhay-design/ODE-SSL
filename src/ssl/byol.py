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

def train_byol_sc(
        online_model, target_model, online_pred_model, energy_model, mlp, train_loader, train_loader_mlp,
        test_loader, lossfunction, lossfunction_mlp, energy_optimizer,
        optimizer, mlp_optimizer, opt_lr_schedular, ema_beta, 
        eval_every, n_epochs, n_epochs_mlp, device_id, eval_id, tsne_name, return_logs=False): 
    

    print(f"### byol-sc Training begins")
    device = torch.device(f"cuda:{device_id}")
    ema = EMA(ema_beta, n_epochs * len(train_loader))
    online_model = online_model.to(device)
    target_model = target_model.to(device)
    online_pred_model = online_pred_model.to(device)
    energy_model = energy_model.to(device)

    global_step = 1

    for epochs in range(n_epochs):
        online_model.train()
        online_pred_model.train()
        energy_model.train()
        en_loss = 0
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, _) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)

            data_all = torch.cat([data, data_cap], dim = 0)

            # _, online_proj = online_model(data_all) # y, z
            output = online_model(data_all) # z
            feat_proj, online_proj = output["features"], output['proj_features']
            online_pred = online_pred_model(online_proj) # q
            with torch.no_grad():
                tar_output = target_model(data_all) # y, z
                target_proj = tar_output['proj_features']

            online_pred_feat, online_pred_feat_cap = online_pred.chunk(2, dim = 0)
            target_proj_feat, target_proj_feat_cap = target_proj.chunk(2, dim = 0)

            feat, feat_cap = feat_proj.chunk(2, dim = 0)

            esample = energy_model.langevin_sampling(feat)
            esample_cap = energy_model.langevin_sampling(feat_cap)
            
            loss_con = lossfunction(online_pred_feat, target_proj_feat_cap.detach()) + \
                        lossfunction(online_pred_feat_cap, target_proj_feat.detach()) + \
                        F.mse_loss(feat, esample.detach()) + F.mse_loss(feat_cap, esample_cap.detach())
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()
            ema(online_model, target_model, global_step)
            global_step += 1

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
    make_tsne_for_dataset(online_model, test_loader, device_id, 'byol-sc', return_logs = return_logs, tsne_name = tsne_name)

    print("### MLP training begins")

    train_mlp(
        online_model, mlp, train_loader_mlp, test_loader, 
        lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
        device_id, eval_id, return_logs = return_logs, algo='byol-sc')

    return online_model


def train_byol(
        online_model, target_model, online_pred_model, mlp, train_loader, train_loader_mlp,
        test_loader, lossfunction, lossfunction_mlp, 
        optimizer, mlp_optimizer, opt_lr_schedular, ema_beta, 
        eval_every, n_epochs, n_epochs_mlp, device_id, eval_id, tsne_name, return_logs=False): 
    

    print(f"### byol Training begins")
    device = torch.device(f"cuda:{device_id}")
    ema = EMA(ema_beta, n_epochs * len(train_loader))
    online_model = online_model.to(device)
    target_model = target_model.to(device)
    online_pred_model = online_pred_model.to(device)

    global_step = 1

    for epochs in range(n_epochs):
        online_model.train()
        online_pred_model.train()
        cur_loss = 0
        len_train = len(train_loader)
        for idx , (data, data_cap, target) in enumerate(train_loader):
            data = data.to(device)
            data_cap = data_cap.to(device)

            data_all = torch.cat([data, data_cap], dim = 0)

            # _, online_proj = online_model(data_all) # y, z
            output = online_model(data_all) # z
            online_proj = output['proj_features']
            online_pred = online_pred_model(online_proj) # q
            with torch.no_grad():
                tar_output = target_model(data_all) # y, z
                target_proj = tar_output['proj_features']

            online_pred_feat, online_pred_feat_cap = online_pred.chunk(2, dim = 0)
            target_proj_feat, target_proj_feat_cap = target_proj.chunk(2, dim = 0)
            
            loss_con = lossfunction(online_pred_feat, target_proj_feat_cap.detach()) + \
                        lossfunction(online_pred_feat_cap, target_proj_feat.detach())
            
            optimizer.zero_grad()
            loss_con.backward()
            optimizer.step()
            ema(online_model, target_model, global_step)
            global_step += 1

            cur_loss += loss_con.item() / (len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_con=loss_con.item(), GPU = device_id)
        
        opt_lr_schedular.step()
            
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_loss_con: {cur_loss:.3f}")

    print("### TSNE starts")
    make_tsne_for_dataset(online_model, test_loader, device_id, 'byol', return_logs = return_logs, tsne_name = tsne_name)

    print("### MLP training begins")

    train_mlp(
        online_model, mlp, train_loader_mlp, test_loader, 
        lossfunction_mlp, mlp_optimizer, n_epochs_mlp, eval_every,
        device_id, eval_id, return_logs = return_logs, algo='byol')

    return online_model