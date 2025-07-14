import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names*")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_state = torch.load('models/best_model_state.pt', map_location=device)
model_metadata = joblib.load('models/model_metadata.joblib')
scaler = joblib.load('models/scaler.joblib')
le = joblib.load('models/label_encoder.joblib')

class Net(nn.Module):
    def __init__(self, input_size, num_classes, dropout, hidden_dims, batchnorm):
        super().__init__()
        layers = []
        in_dim = input_size
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model_nn = Net(
    input_size=model_metadata['input_size'],
    num_classes=model_metadata['num_classes'],
    dropout=model_metadata['dropout'],
    hidden_dims=model_metadata['hidden_dims'],
    batchnorm=model_metadata['batchnorm']
).to(device)
model_nn.load_state_dict(model_state)

def predict_pose(keypoints_norm):
    x_input = keypoints_norm.flatten().reshape(1, -1)
    x_scaled = scaler.transform(x_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
    model_nn.eval()
    with torch.no_grad():
        logits = model_nn(x_tensor)
        prob = F.softmax(logits, dim=1)
        predicted_idx = logits.argmax(dim=1).item()
        predicted_prob = prob[0, predicted_idx].item()
        predicted_pose = le.inverse_transform([predicted_idx])[0]
    return predicted_pose, predicted_prob