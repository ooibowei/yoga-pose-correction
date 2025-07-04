import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

x_train_aug = pd.read_parquet('data/processed/x_train_aug.parquet')
x_val = pd.read_parquet('data/processed/x_val.parquet')
y_train_aug = pd.read_parquet('data/processed/y_train_aug.parquet')['label']
y_val = pd.read_parquet('data/processed/y_val.parquet')['label']

scaler = StandardScaler().fit(x_train_aug)
le = LabelEncoder().fit(y_train_aug)

model_logistic = LogisticRegression(max_iter=1000).fit(scaler.transform(x_train_aug), y_train_aug)
yhat_logistic = model_logistic.predict(scaler.transform(x_val))
print('Logistic Regression F1', f1_score(y_val, yhat_logistic, average='macro'))

model_xgb = XGBClassifier().fit(x_train_aug, le.transform(y_train_aug))
yhat_xgb = le.inverse_transform(model_xgb.predict(x_val))
print('XGBoost F1', f1_score(y_val, yhat_xgb, average='macro'))

model_rf = RandomForestClassifier().fit(x_train_aug, y_train_aug)
yhat_rf = model_rf.predict(x_val)
print('Random Forest F1', f1_score(y_val, yhat_rf, average='macro'))

class Net(nn.Module):
    def __init__(self, input_size=132, num_classes=82):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.output(x)

x_train_tensor = torch.tensor(scaler.transform(x_train_aug), dtype=torch.float32)
x_val_tensor = torch.tensor(scaler.transform(x_val), dtype=torch.float32)
y_train_tensor = torch.tensor(le.transform(y_train_aug), dtype=torch.long)
y_val_tensor = torch.tensor(le.transform(y_val), dtype=torch.long)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_nn = Net(input_size=132, num_classes=82).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_nn.parameters())

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
    print(f'Train Loss: {loss_train/size_train:.6f}, Train F1 {f1_train:.4f}')

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
    print(f'Val Loss {loss_test/size_test:.6f}, Val F1 {f1_test:.4f}')
    return labels_test, yhat_test

epochs = 30
for t in range(epochs):
    print(f"-------------------------------\nEpoch {t+1}\n-------------------------------")
    train_loop(train_loader, model_nn, loss_fn, optimizer)
    test_loop(val_loader, model_nn, loss_fn)

labels_val, yhat_val = test_loop(val_loader, model_nn, loss_fn)
print('FFN F1:', f1_score(labels_val, yhat_val, average='macro'))