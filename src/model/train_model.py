import joblib
import optuna
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

x_train_aug = pd.read_parquet('data/processed/x_train_aug.parquet')
x_val = pd.read_parquet('data/processed/x_val.parquet')
y_train_aug = pd.read_parquet('data/processed/y_train_aug.parquet')['label']
y_val = pd.read_parquet('data/processed/y_val.parquet')['label']

scaler = StandardScaler().fit(x_train_aug)
le = LabelEncoder().fit(y_train_aug)

x_train_tensor = torch.tensor(scaler.transform(x_train_aug), dtype=torch.float32)
x_val_tensor = torch.tensor(scaler.transform(x_val), dtype=torch.float32)
y_train_tensor = torch.tensor(le.transform(y_train_aug), dtype=torch.long)
y_val_tensor = torch.tensor(le.transform(y_val), dtype=torch.long)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_size=132, num_classes=82, dropout=0.2, hidden_dims=[256, 128]):
        super(Net, self).__init__()

        layers = []
        in_dim = input_size
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    loss_train = 0
    size_train = len(dataloader.dataset)
    labels_train = []
    yhat_train = []
    for x_b, y_b in dataloader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        logits = model(x_b)
        loss = loss_fn(logits, y_b)
        loss_train += loss.item()
        loss.backward() # Backprop to obtain gradients
        optimizer.step()
        optimizer.zero_grad()

        yhat_b = logits.argmax(dim=1).cpu().numpy()
        yhat_train.extend(yhat_b)
        labels_train.extend(y_b.cpu().numpy())

    f1_train = f1_score(labels_train, yhat_train, average='macro')
    return loss_train/size_train, f1_train

def test_loop(dataloader, model, loss_fn):
    model.eval()
    loss_test = 0
    size_test = len(dataloader.dataset)
    labels_test = []
    yhat_test = []
    with torch.no_grad():
        for x_b, y_b in dataloader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = model(x_b)
            loss = loss_fn(logits, y_b)
            loss_test += loss.item()

            yhat_b = logits.argmax(dim=1).cpu().numpy()
            yhat_test.extend(yhat_b)
            labels_test.extend(y_b.cpu().numpy())

    f1_test = f1_score(labels_test, yhat_test, average='macro')
    return loss_test/size_test, f1_test

def objective(trial):
    # Hyperparameters
    factor = trial.suggest_categorical('factor', [0.1, 0.2, 0.3, 0.4, 0.5])
    patience = trial.suggest_int('patience', 3, 10)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    beta1 = trial.suggest_float('beta1', 0.8, 0.99)
    beta2 = trial.suggest_float('beta2', 0.9, 0.9999)
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 5)
    hidden_dims = [trial.suggest_categorical(f"hidden_dim_{i}", [64, 128, 256, 512]) for i in range(num_hidden_layers)]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    model_nn = Net(input_size=132, num_classes=82, dropout=dropout_rate, hidden_dims=hidden_dims).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, min_lr=1e-6)

    max_epochs = 100
    best_val_f1 = 0
    early_stop_patience = 10
    patience_count = 0

    train_losses, val_losses, train_f1s, val_f1s, lrs = [], [], [], [], []

    for epoch in range(max_epochs):
        train_loss, train_f1 = train_loop(train_loader, model_nn, loss_fn, optimizer)
        val_loss, val_f1 = test_loop(val_loader, model_nn, loss_fn)
        scheduler.step(val_f1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        lrs.append(optimizer.param_groups[0]['lr'])

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = deepcopy(model_nn.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count == early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    trial.set_user_attr("train_losses", train_losses)
    trial.set_user_attr("val_losses", val_losses)
    trial.set_user_attr("train_f1s", train_f1s)
    trial.set_user_attr("val_f1s", val_f1s)
    trial.set_user_attr("lrs", lrs)
    trial.set_user_attr("epochs_trained", len(train_losses))

    model_nn.load_state_dict(best_model_state)
    final_val_loss, final_val_f1 = test_loop(val_loader, model_nn, loss_fn)
    print(f'Final Val F1: {final_val_f1:.4f}')
    return final_val_f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, n_jobs=4)
joblib.dump(study, "models/study.pkl")
optuna_bundle = {'best_value': study.best_value, 'best_params': study.best_params, 'study': study, 'user_attrs': study.user_attrs}
joblib.dump(optuna_bundle, 'models/optuna_bundle.joblib')
print("Best F1 Score:", study.best_value)
print("Best Parameters:")
for k, v in study.best_params.items():
    print(f"   {k}: {v}")
"""
Best F1 = 0.875
Best parameters  
   factor: 0.1
   patience: 7
   lr: 0.000985655800431002
   batch_size: 128
   dropout_rate: 0.43576903523576654
   weight_decay: 1.008463977584377e-06
   beta1: 0.9401132856796786
   beta2: 0.9407801128814276
   num_hidden_layers: 2
   hidden_dim_0: 256
   hidden_dim_1: 512
"""

epochs = range(study.best_trial.user_attrs["epochs_trained"])
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, study.best_trial.user_attrs["train_losses"], label='Train Loss')
plt.plot(epochs, study.best_trial.user_attrs["val_losses"], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, study.best_trial.user_attrs["train_f1s"], label='Train F1')
plt.plot(epochs, study.best_trial.user_attrs["val_f1s"], label='Val F1')
plt.title("F1 Score Curve")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()

plt.tight_layout()
plt.savefig("reports/images/loss_f1.png")
plt.show()

# Train and save final model
best_params = study.best_params
train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=True)
hidden_dims = [best_params[f"hidden_dim_{i}"] for i in range(best_params["num_hidden_layers"])]
model_nn = Net(input_size=132, num_classes=82, dropout=best_params["dropout_rate"], hidden_dims=hidden_dims).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"], betas=(best_params["beta1"], best_params["beta2"]))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=best_params["factor"], patience=best_params["patience"], min_lr=1e-6)

max_epochs = 100
best_val_f1 = 0
early_stop_patience = 10
patience_count = 0
for epoch in range(max_epochs):
    train_loss, train_f1 = train_loop(train_loader, model_nn, loss_fn, optimizer)
    val_loss, val_f1 = test_loop(val_loader, model_nn, loss_fn)
    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = deepcopy(model_nn.state_dict())
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= early_stop_patience:
            print(f'Early Stopping {epoch}')
            break

model_nn.load_state_dict(best_model_state)
val_loss, val_f1 = test_loop(val_loader, model_nn, loss_fn)
print(f"Best validation F1: {val_f1:.4f}")

torch.save(best_model_state, 'models/best_model_state.pt')
model_metadata = {"input_size": 132, "num_classes": 82, "dropout": best_params["dropout_rate"], "hidden_dims": hidden_dims}
joblib.dump(model_metadata, 'models/model_metadata.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(le, 'models/label_encoder.joblib')