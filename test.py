import torch
import torch.nn as nn
import torch.optim as optim

def evaluate(model, mlp, loader, device, return_logs=False, algo=None):
    model.eval()
    mlp.eval()
    correct = 0;samples =0
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            output = model(x, test=True)
            feats = output["features"]
            scores = mlp(feats)

            predict_prob = F.softmax(scores,dim=1)
            _,predictions = predict_prob.max(1)

            correct += (predictions == y).sum()
            samples += predictions.size(0)
        
            if return_logs:
                progress(idx+1,loader_len)
                # print('batches done : ',idx,end='\r')
        accuracy = round(float(correct / samples), 3)
    return accuracy 

def train_mlp(
    model, mlp, train_loader, test_loader, 
    lossfunction, mlp_optimizer, n_epochs, eval_every,
    device_id, eval_id, return_logs=False, mlp_schedular=None):

    tval = {'trainacc':[],"trainloss":[], "testacc":[]}
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)
    mlp = mlp.to(device)
    for epochs in range(n_epochs):
        model.eval()
        mlp.train()
        curacc = 0
        cur_mlp_loss = 0
        len_train = len(train_loader)
        for idx , (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            with torch.no_grad():
                output = model(data, test=True)
                feats = output["features"]
            scores = mlp(feats.detach())      
            
            loss_sup = lossfunction(scores, target)

            mlp_optimizer.zero_grad()
            loss_sup.backward()
            mlp_optimizer.step()

            cur_mlp_loss += loss_sup.item() / (len_train)
            scores = F.softmax(scores,dim = 1)
            _,predicted = torch.max(scores,dim = 1)
            correct = (predicted == target).sum()
            samples = scores.shape[0]
            curacc += correct / (samples * len_train)
            
            if return_logs:
                progress(idx+1,len(train_loader), loss_sup=loss_sup.item(), GPU = device_id)
        
        if mlp_schedular is not None:
            mlp_schedular.step()
        
        if epochs % eval_every == 0 and device_id == eval_id:
            cur_test_acc = evaluate(model, mlp, test_loader, device, return_logs, algo=algo)
            tval["testacc"].append(float(cur_test_acc))
            print(f"[GPU{device_id}] Test Accuracy at epoch: {epochs}: {cur_test_acc}")
      
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_mlp_loss))
        
        print(f"[GPU{device_id}] epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss_sup: {cur_mlp_loss:.3f}")
    
    if device_id == eval_id:
        final_test_acc = evaluate(model, mlp, test_loader, device, return_logs, algo=algo)
        print(f"[GPU{device_id}] Final Test Accuracy: {final_test_acc}")

    return mlp, tval

def train_linear_probe(
    train_loader, 
    test_loader, 
    feature_dim, 
    num_classes, 
    device='cuda', 
    epochs=100
):
    """
    Trains a linear probe and grid-searches LR and Weight Decay for optimal performance.
    """
    # Standard sweep grids for linear probing
    learning_rates = [0.1, 1.0]
    weight_decays = [1e-6, 1e-4, 0.0]
    
    best_acc = 0.0
    best_model_state = None
    best_hparams = {}

    criterion = nn.CrossEntropyLoss()

    print(f"Starting Hyperparameter Sweep on {device}...")
    
    for lr, wd in itertools.product(learning_rates, weight_decays):
        model = LinearProbe(feature_dim, num_classes).to(device)
        
        # Best Practice: Do not apply weight decay to the bias term
        parameters = [
            {'params': [model.fc.weight], 'weight_decay': wd},
            {'params': [model.fc.bias], 'weight_decay': 0.0}
        ]
        
        # SGD with momentum is standard and usually outperforms Adam for linear probes
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            model.train()
            for features, targets in train_loader:
                features, targets = features.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            scheduler.step()

        # Evaluate on validation set
        val_acc = evaluate(model, test_loader, device)
        
        print(f"LR: {lr:5.3f} | WD: {wd:7.6f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_hparams = {'lr': lr, 'wd': wd}

    print("-" * 30)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Optimal Hyperparameters: LR={best_hparams['lr']}, WD={best_hparams['wd']}")
    
    # Load best weights into the final model
    optimal_model = LinearProbe(feature_dim, num_classes).to(device)
    optimal_model.load_state_dict(best_model_state)
    
    return optimal_model, best_hparams

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    return 100 * correct / total


if __name__ == "__main__":
    pass 