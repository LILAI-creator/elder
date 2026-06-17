import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3

DATASET_DIR = r"D:\my_datasets\Le2i\processed"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


class FallDataset(Dataset):
    def __init__(self, X, risk, time):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.risk = torch.tensor(risk, dtype=torch.float32)
        self.time = torch.tensor(time, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.risk[idx], self.time[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size=171, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.risk_head = nn.Linear(hidden_size, 1)
        self.time_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        risk = torch.sigmoid(self.risk_head(out))
        time = torch.relu(self.time_head(out))
        return risk.squeeze(-1), time.squeeze(-1)


def train_one_epoch(model, loader, criterion_risk, criterion_time, optimizer, lambda_time):
    model.train()
    total_loss = 0
    total_risk_loss = 0
    total_time_loss = 0
    n = 0
    for X_batch, risk_batch, time_batch in loader:
        X_batch = X_batch.to(DEVICE)
        risk_batch = risk_batch.to(DEVICE)
        time_batch = time_batch.to(DEVICE)

        pred_risk, pred_time = model(X_batch)
        loss_risk = criterion_risk(pred_risk, risk_batch)

        fall_mask = risk_batch > 0
        if fall_mask.sum() > 0:
            loss_time = criterion_time(pred_time[fall_mask], time_batch[fall_mask])
        else:
            loss_time = torch.tensor(0.0, device=DEVICE)
        loss = loss_risk + lambda_time * loss_time

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(risk_batch)
        total_risk_loss += loss_risk.item() * len(risk_batch)
        total_time_loss += loss_time.item() * len(risk_batch)
        n += len(risk_batch)

    return total_loss / n, total_risk_loss / n, total_time_loss / n


def evaluate(model, loader, criterion_risk, criterion_time, lambda_time):
    model.eval()
    total_loss = 0
    total_risk_loss = 0
    total_time_loss = 0
    all_pred_risk = []
    all_pred_time = []
    all_risk = []
    all_time = []
    n = 0
    with torch.no_grad():
        for X_batch, risk_batch, time_batch in loader:
            X_batch = X_batch.to(DEVICE)
            risk_batch = risk_batch.to(DEVICE)
            time_batch = time_batch.to(DEVICE)

            pred_risk, pred_time = model(X_batch)
            loss_risk = criterion_risk(pred_risk, risk_batch)

            fall_mask = risk_batch > 0
            if fall_mask.sum() > 0:
                loss_time = criterion_time(pred_time[fall_mask], time_batch[fall_mask])
            else:
                loss_time = torch.tensor(0.0, device=DEVICE)
            loss = loss_risk + lambda_time * loss_time

            total_loss += loss.item() * len(risk_batch)
            total_risk_loss += loss_risk.item() * len(risk_batch)
            total_time_loss += loss_time.item() * len(risk_batch)
            n += len(risk_batch)

            all_pred_risk.extend(pred_risk.cpu().numpy())
            all_pred_time.extend(pred_time.cpu().numpy())
            all_risk.extend(risk_batch.cpu().numpy())
            all_time.extend(time_batch.cpu().numpy())

    return (
        total_loss / n,
        total_risk_loss / n,
        total_time_loss / n,
        np.array(all_pred_risk),
        np.array(all_pred_time),
        np.array(all_risk),
        np.array(all_time),
    )


def main():
    print(f"Device: {DEVICE}")

    X = np.load(os.path.join(DATASET_DIR, "x.npy"))
    risk = np.load(os.path.join(DATASET_DIR, "risk.npy")).squeeze(-1)
    time = np.load(os.path.join(DATASET_DIR, "time.npy")).squeeze(-1)
    print(f"Loaded X: {X.shape}, risk: {risk.shape}, time: {time.shape}")

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    risk_train, risk_test = risk[train_idx], risk[test_idx]
    time_train, time_test = time[train_idx], time[test_idx]

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Normalized: mean≈{X_train.mean():.4f}, std≈{X_train.std():.4f}")

    train_loader = DataLoader(
        FallDataset(X_train, risk_train, time_train), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        FallDataset(X_test, risk_test, time_test), batch_size=BATCH_SIZE, shuffle=False
    )

    lambda_time = 0.1
    model = LSTMModel().to(DEVICE)
    criterion_risk = nn.MSELoss()
    criterion_time = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n{'Epoch':>5} {'Loss':>10} {'RiskL':>10} {'TimeL':>10} {'tLoss':>10} {'tRiskL':>10} {'tTimeL':>10}")
    print("-" * 70)

    best_test_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_risk_l, train_time_l = train_one_epoch(
            model, train_loader, criterion_risk, criterion_time, optimizer, lambda_time
        )
        test_loss, test_risk_l, test_time_l, _, _, _, _ = evaluate(
            model, test_loader, criterion_risk, criterion_time, lambda_time
        )
        scheduler.step()
        print(
            f"{epoch:>5} {train_loss:>10.4f} {train_risk_l:>10.4f} {train_time_l:>10.4f}"
            f" {test_loss:>10.4f} {test_risk_l:>10.4f} {test_time_l:>10.4f}"
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_multitask.pt"))
            np.savez(
                os.path.join(MODEL_DIR, "norm_params.npz"),
                mean=mean.squeeze(),
                std=std.squeeze(),
            )

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "lstm_multitask.pt"), map_location=DEVICE))
    _, _, _, pred_risk, pred_time, true_risk, true_time = evaluate(
        model, test_loader, criterion_risk, criterion_time, lambda_time
    )

    risk_mae = np.abs(pred_risk - true_risk).mean()
    time_mae = np.abs(pred_time - true_time).mean()
    fall_mask = true_risk > 0
    normal_mask = true_risk == 0
    print(f"\n=== Test Results ===")
    print(f"Risk MAE: {risk_mae:.4f}")
    print(f"Time MAE: {time_mae:.4f}")
    if fall_mask.sum() > 0:
        print(f"Fall Risk MAE: {np.abs(pred_risk[fall_mask] - true_risk[fall_mask]).mean():.4f}")
        print(f"Fall Time MAE: {np.abs(pred_time[fall_mask] - true_time[fall_mask]).mean():.4f}")
    if normal_mask.sum() > 0:
        print(f"Normal Risk MAE: {np.abs(pred_risk[normal_mask] - true_risk[normal_mask]).mean():.4f}")

    print(f"\nModel saved to {os.path.join(MODEL_DIR, 'lstm_multitask.pt')}")
    print(f"Norm params saved to {os.path.join(MODEL_DIR, 'norm_params.npz')}")


if __name__ == "__main__":
    main()
