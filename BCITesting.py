import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mne
from sklearn.metrics import classification_report, confusion_matrix

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

path = r"BCICIV_2a_gdf"

subjects = np.empty(9, dtype=object)

FS = 250
WIN = 4 * FS

for i in range(9):
    print("Subject " + str(i) + " started")

    gdf_path = path + f"\\A{i+1:02d}T.gdf"

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

    EEG = raw.get_data(picks=np.arange(22)).T.astype(np.float32)

    events, _ = mne.events_from_annotations(raw, verbose=False)

    rejected_pos = set(events[events[:, 2] == 1][:, 0])

    data = []
    labels = []

    for pos, _, code in events:
        if (code not in (7, 8)) or (pos in rejected_pos):
            continue

        end = pos + WIN
        if end <= EEG.shape[0]:
            seg = EEG[pos:end]
            seg = (seg-seg.mean(0)) / (seg.std(0) + 0.000001)
            data.append(seg)
            labels.append(code - 7)

    X = np.stack(data, axis=0)
    y = np.array(labels, dtype=np.int64)

    subjects[i] = pd.DataFrame({
        "X": [X],
        "y": [y]
    })

    print("Subject " + str(i) + " data preprocessing complete")

class Data(Dataset):
    def __init__(self, X, y):
        X = np.transpose(X, (0, 2, 1))
        X = X[:, None, :, :]

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_classes=2, samples=1000,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, samples)
            out = self.conv3(self.conv2(self.conv1(dummy)))
            self.flat_dim = out.numel()

        self.classifier = nn.Linear(self.flat_dim, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        return self.classifier(x)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0,0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return correct / total

def train(subjects, epochs=75, batch_size=32, learning_rate=0.01, weight_decay=0.005, device=None):
    device = "cuda"

    X_train = np.concatenate([subjects[i].loc[0, "X"] for i in range(7)], axis=0)
    y_train = np.concatenate([subjects[i].loc[0, "y"] for i in range(7)], axis=0)

    X_test = np.concatenate([subjects[7].loc[0, "X"], subjects[8].loc[0, "X"]])
    y_test = np.concatenate([subjects[7].loc[0, "y"], subjects[8].loc[0, "y"]])
    
    train_ds = Data(X_train, y_train)
    test_ds = Data(X_test, y_test)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = EEGNet(
        samples = X_train.shape[1]
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    criteria = nn.CrossEntropyLoss()

    max_acc = 0

    target_names = ['left', 'right']

    for e in range(epochs):
        model.train()
        tot_loss = 0.0
        correct = 0
        total = 0

        y_true = []
        y_pred = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criteria(logits, yb)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            tot_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

        accuracy = correct / total
        avg_loss = tot_loss / total
        test_acc = evaluate(model, test_loader, device)

        if (max_acc < test_acc):
            max_acc = test_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_channels": 22,
                    "samples": X.shape[1],
                },
                "eegnet_leftright.pt"
            )

        print("Epoch " + str(e+1) + ": Loss = " + str(avg_loss) + ", Train_Acc = " + str(accuracy) + ", Test_Acc = " + str(test_acc))
        print(classification_report(y_true, y_pred, target_names=target_names))
        print(confusion_matrix(y_true, y_pred))

    print("Max Accuracy = " + str(max_acc))
    return model

model = train(subjects)