# basic_cnn_train.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set GPU device (GPU 0 for this script)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ======================
# Configuration settings
# ======================
root_folder = '/home/aa3038@students.ad.unt.edu/DL/output/'
csv_path = '/home/aa3038@students.ad.unt.edu/DL/fMRI_3_24_2025.csv'
batch_size = 8
num_epochs = 10
lr = 0.001

# =======================
# Load and filter metadata
# =======================
df = pd.read_csv(csv_path)

def has_nifti(img_id):
    folder = os.path.join(root_folder, str(img_id))
    if not os.path.isdir(folder):
        return False
    for f in os.listdir(folder):
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            return True
    return False

df['has_nii'] = df['Image Data ID'].apply(has_nifti)
df = df[df['has_nii']].reset_index(drop=True)
df['label'] = df['Group'].apply(lambda x: 0 if x in ['CN', 'SMC'] else 1)

# ==========================
# Volume loading + resizing
# ==========================
def load_volume(path):
    for f in os.listdir(path):
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            img = nib.load(os.path.join(path, f))
            data = img.get_fdata()
            return np.mean(data, axis=3)
    raise FileNotFoundError("No NIfTI file found.")

def resize_volume(img, size=(64,64,64)):
    import scipy.ndimage
    zoom = [s / float(img.shape[i]) for i, s in enumerate(size)]
    return scipy.ndimage.zoom(img, zoom, order=1)

# ====================
# Custom PyTorch Dataset
# ====================
class FMRIDataset(torch.utils.data.Dataset):
    def __init__(self, df, root):
        self.df = df
        self.root = root
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vol = load_volume(os.path.join(self.root, str(row['Image Data ID'])))
        vol = resize_volume(vol)
        vol = (vol - vol.mean()) / (vol.std() + 1e-5)
        vol = np.expand_dims(vol, 0)
        return torch.tensor(vol, dtype=torch.float32), torch.tensor(int(row['label']))

# ====================
# Train/test split + loaders
# ====================
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_loader = DataLoader(FMRIDataset(train_df, root_folder), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(FMRIDataset(test_df, root_folder), batch_size=batch_size)

# ====================
# Define Basic 3D CNN Model
# ====================
class Basic3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(32 * 16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 2)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ====================
# Initialize model, optimizer, loss
# ====================
model = Basic3DCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
metrics = []

# ====================
# Training Loop
# ====================
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")
    metrics.append({'epoch': epoch+1, 'loss': avg_loss, 'accuracy': acc})

# ====================
# Evaluation - Classification Report
# ====================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"])
print("✅ Classification Report:\n")
print(report)

# ====================
# Save results
# ====================
output_dir = "/home/aa3038@students.ad.unt.edu/DL/output_csv"
pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "basic_cnn_metrics.csv"), index=False)
with open(os.path.join(output_dir, "basic_cnn_classification_report.txt"), "w") as f:
    f.write(report)

print("✅ Metrics saved to basic_cnn_metrics.csv")
print("✅ Classification report saved to basic_cnn_classification_report.txt")
