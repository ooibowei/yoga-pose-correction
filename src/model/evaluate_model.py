import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

model_state = torch.load('models/best_model_state.pt', map_location=torch.device('cpu'))
model_metadata = joblib.load('models/model_metadata.joblib')
scaler = joblib.load('models/scaler.joblib')
le = joblib.load('models/label_encoder.joblib')

x_test = pd.read_parquet('data/processed/x_test.parquet')
y_test = pd.read_parquet('data/processed/y_test.parquet')['label']
x_test_tensor = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
y_test_tensor = torch.tensor(le.transform(y_test), dtype=torch.long)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    return loss_test/size_test, f1_test, labels_test, yhat_test

class Net(nn.Module):
    def __init__(self, input_size, num_classes, dropout, hidden_dims):
        super().__init__()
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

model_nn = Net(
    input_size=model_metadata['input_size'],
    num_classes=model_metadata['num_classes'],
    dropout=model_metadata['dropout'],
    hidden_dims=model_metadata['hidden_dims'],
)
model_nn.load_state_dict(model_state)
model_nn.eval()
loss_fn = nn.CrossEntropyLoss()
loss_test, f1_test, labels_test, yhat_test = test_loop(test_loader, model_nn, loss_fn)
print(f"Test F1 Score: {f1_test:.4f}") # Final test F1 score is 0.867

# Lowest F1 scores
report = classification_report(labels_test, yhat_test, target_names=le.classes_, output_dict=True)
class_f1 = {pose: metrics["f1-score"] for pose, metrics in report.items() if pose in le.classes_}
class_f1 = pd.Series(class_f1)
lowest_f1 = class_f1.sort_values().head(10).reset_index()
lowest_f1.columns = ['Pose', 'F1 Score']
lowest_f1['Pose'] = lowest_f1['Pose'].str.slice(0, 15)
plt.figure(figsize=(12, 6))
sns.barplot(data=lowest_f1, x='F1 Score', y='Pose')
plt.title(f"Bottom 10 Poses by F1 Score")
plt.xlabel("F1 Score")
plt.ylabel("Pose")
plt.tight_layout()
plt.savefig("reports/images/lowest_f1_poses.png")
plt.show()

# Misclassified Poses
cm = confusion_matrix(labels_test, yhat_test)
true_counts = cm.sum(axis=1).reshape((82, 1))
cm_norm = cm/true_counts
np.fill_diagonal(cm_norm, 0)
cm_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
misclassified_poses = cm_df.unstack().sort_values(ascending=False).head(10).rename_axis(['Predicted', 'True']).reset_index(name='Normalised')
misclassified_poses['True'] = misclassified_poses['True'].str[:15]
misclassified_poses['Predicted'] = misclassified_poses['Predicted'].str[:15]
misclassified_poses['Pair'] = (misclassified_poses['True'] + ', ' + misclassified_poses['Predicted'])

plt.figure(figsize=(12, 6))
sns.barplot(data=misclassified_poses, x='Normalised', y='Pair')
plt.title("Top 10 Normalized Misclassified Pose Pairs")
plt.xlabel("Normalized Confusion Rate")
plt.ylabel("True, Predicted")
plt.tight_layout()
plt.savefig("reports/images/misclassified_poses.png")
plt.show()

# Misclassified Noose -> Garland
img_true = plt.imread("data/raw/Noose_Pose_or_Pasasana_/Noose_Pose_or_Pasasana__image_5.jpg")
img_pred = plt.imread('data/raw/Garland_Pose_or_Malasana_/Garland_Pose_or_Malasana__image_113.jpg')
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(img_true)
axs[0].axis('off')
axs[0].set_title("True Pose: Noose Pose")
axs[1].imshow(img_pred)
axs[1].axis('off')
axs[1].set_title("Misclassified as: Garland Pose")
plt.suptitle("Comparison of True and Misclassified Poses")
plt.tight_layout()
plt.savefig('reports/images/noose_garland.png')
plt.show()

# Misclassified Heron -> Boat
img_true = plt.imread("data/raw/Heron_Pose_or_Krounchasana_/Heron_Pose_or_Krounchasana__image_1.jpg")
img_pred = plt.imread('data/raw/Boat_Pose_or_Paripurna_Navasana_/Boat_Pose_or_Paripurna_Navasana__image_11.jpg')
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(img_true)
axs[0].axis('off')
axs[0].set_title("True Pose: Heron Pose")
axs[1].imshow(img_pred)
axs[1].axis('off')
axs[1].set_title("Misclassified as: Boat Pose")
plt.suptitle("Comparison of True and Misclassified Poses")
plt.tight_layout()
plt.savefig('reports/images/heron_boat.png')
plt.show()