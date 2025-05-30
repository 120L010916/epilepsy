import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset
from models.CNN import SeizureCNN


def load_data(npz_path):
    data = np.load(npz_path)
    X = data['F']  # (N, 2, 36)
    y = data['y']
    X = (X - X.mean()) / X.std()
    X_tensor = torch.tensor(X).float().unsqueeze(1)  # (N, 1, 2, 36)
    y_tensor = torch.tensor(y).long()
    return X_tensor, y_tensor


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_data(args.input)

    dataset = TensorDataset(X, y)
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    all_acc = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n📂 Fold {fold+1}/{args.k_folds}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size)

        model = SeizureCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        all_acc.append(val_acc)

    print(f"\n✅ Cross-validation 平均准确率：{np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SeizureCNN with cross-validation.")
    parser.add_argument('--input', type=str, default='data/features/chb01/features.npz', help='Path to input .npz file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()
    main(args)
