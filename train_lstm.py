import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


class FallDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=34,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        logits = model(X_batch)
        loss = criterion(logits, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(Y_batch)
        pred = logits.argmax(dim=1)
        correct += (pred == Y_batch).sum().item()
        total += len(Y_batch)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, Y_batch)

            total_loss += loss.item() * len(Y_batch)
            pred = logits.argmax(dim=1)
            correct += (pred == Y_batch).sum().item()
            total += len(Y_batch)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(Y_batch.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    print(f"Device: {DEVICE}")

    X = np.load(os.path.join(DATASET_DIR, "X.npy"))
    Y = np.load(os.path.join(DATASET_DIR, "Y.npy"))
    print(f"Loaded X: {X.shape}, Y: {Y.shape}")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Z-score normalization on train, apply to test
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    print(f"Normalized: mean≈{X_train.mean():.4f}, std≈{X_train.std():.4f}")

    train_loader = DataLoader(
        FallDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        FallDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    # Class weights
    n_total = len(Y_train)
    n_safe = (Y_train == 0).sum()
    n_danger = (Y_train == 1).sum()
    w0 = n_total / (2 * n_safe)
    w1 = n_total / (2 * n_danger)
    print(f"Class weights: Safe={w0:.4f}, Danger={w1:.4f}")

    model = LSTMClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>10} {'TestLoss':>10} {'TestAcc':>10}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step()
        print(
            f"{epoch:>5} {train_loss:>10.4f} {train_acc:>10.4f} {test_loss:>10.4f} {test_acc:>10.4f}"
        )

    # Final evaluation
    _, _, y_pred, y_true = evaluate(model, test_loader, criterion)

    label_names = ["Safe", "Danger"]
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    print(f"{'':>10} {'PredSafe':>10} {'PredDanger':>10}")
    print(f"{'TrueSafe':>10} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"{'TrueDanger':>10} {cm[1][0]:>10} {cm[1][1]:>10}")

    # Save model + norm params
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lstm_baseline.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    norm_path = os.path.join(MODEL_DIR, "norm_params.npz")
    np.savez(norm_path, mean=mean.squeeze(), std=std.squeeze())
    print(f"Norm params saved to {norm_path}")


if __name__ == "__main__":
    main()
