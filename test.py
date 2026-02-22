import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import itertools
import copy

class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_classes, normalize_features=True):
        super().__init__()
        self.normalize_features = normalize_features
        self.fc = nn.Linear(feature_dim, num_classes)
        
        # Standard initialization for linear probes
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        if self.normalize_features:
            # L2 normalization is highly recommended for learned features
            x = torch.nn.functional.normalize(x, dim=1, p=2)
        return self.fc(x)

def train_linear_probe(
    train_loader, 
    val_loader, 
    feature_dim, 
    num_classes, 
    device='cuda', 
    epochs=100
):
    """
    Trains a linear probe and grid-searches LR and Weight Decay for optimal performance.
    """
    # Standard sweep grids for linear probing
    learning_rates = [0.01, 0.1, 0.3, 1.0, 3.0]
    weight_decays = [1e-6, 1e-5, 1e-4, 1e-3, 0.0]
    
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
        val_acc = evaluate(model, val_loader, device)
        
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

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dummy data (replace with your actual extracted feature tensors)
    N_TRAIN, N_VAL, DIM, CLASSES = 5000, 1000, 512, 10
    
    train_features = torch.randn(N_TRAIN, DIM)
    train_labels = torch.randint(0, CLASSES, (N_TRAIN,))
    val_features = torch.randn(N_VAL, DIM)
    val_labels = torch.randint(0, CLASSES, (N_VAL,))

    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_features, val_labels), batch_size=256, shuffle=False)

    best_model, best_params = train_linear_probe(
        train_loader, val_loader, 
        feature_dim=DIM, num_classes=CLASSES, 
        device=device, epochs=50
    )