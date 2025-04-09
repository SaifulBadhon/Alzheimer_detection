import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from monai.networks.nets import resnet

# ===============
# Set GPU device (GPU 1)
# ===============
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

# ===============
# Configuration
# ===============
root_folder = '/home/aa3038@students.ad.unt.edu/DL/output'
csv_path = '/home/aa3038@students.ad.unt.edu/DL/fMRI_3_24_2025.csv'
output_dir = '/home/aa3038@students.ad.unt.edu/DL/output_csv'
batch_size = 8
num_epochs = 10
lr = 0.001

# ===============
# Load and filter metadata
# ===============
def has_nifti(img_id):
    folder = os.path.join(root_folder, str(img_id))
    if not os.path.isdir(folder): return False
    for f in os.listdir(folder):
        if f.endswith(".nii") or f.endswith(".nii.gz"):
            return True
    return False

df = pd.read_csv(csv_path)
df['has_nii'] = df['Image Data ID'].apply(has_nifti)
df = df[df['has_nii']].reset_index(drop=True)
df['label'] = df['Group'].apply(lambda x: 0 if x in ['CN', 'SMC'] else 1)
print(f"✅ Total usable samples: {len(df)}")

# ===============
# Volume utilities
# ===============
def load_volume(path):
    for f in os.listdir(path):
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            img = nib.load(os.path.join(path, f))
            return np.mean(img.get_fdata(), axis=3)
    raise FileNotFoundError(f"No NIfTI file found in: {path}")

def resize_volume(img, size=(64, 64, 64)):
    import scipy.ndimage
    zoom = [s / float(img.shape[i]) for i, s in enumerate(size)]
    return scipy.ndimage.zoom(img, zoom, order=1)

# ===============
# PyTorch Dataset
# ===============
class FMRIDataset(Dataset):
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

# ===============
# Data split
# ===============
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_loader = DataLoader(FMRIDataset(train_df, root_folder), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(FMRIDataset(test_df, root_folder), batch_size=batch_size)

# ===============
# Define ResNet3D model
# ===============
model = resnet.resnet18(
    spatial_dims=3,
    n_input_channels=1,
    num_classes=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
metrics = []

# ===============
# Training loop
# ===============
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

# ===============
# Evaluation - Classification Report & Confusion Matrix
# ===============
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"])
cm = confusion_matrix(all_labels, all_preds)

print("✅ Classification Report:\n")
print(report)
print("✅ Confusion Matrix:\n")
print(cm)

# ===============
# Save metrics
# ===============
os.makedirs(output_dir, exist_ok=True)

# Save training metrics
pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "resnet3d_metrics.csv"), index=False)

# Save report and confusion matrix
with open(os.path.join(output_dir, "resnet3d_classification_report.txt"), "w") as f:
    f.write("Classification Report:\n")
    f.write(report + "\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))

print("✅ Metrics saved to resnet3d_metrics.csv")
print("✅ Classification report and confusion matrix saved to resnet3d_classification_report.txt")
