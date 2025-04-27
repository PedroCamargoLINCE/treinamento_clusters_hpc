#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import argparse
import zipfile
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Treinamento de GRU/XGBoost por cluster")
parser.add_argument("--cluster_id", type=int, required=True, help="ID do cluster a ser treinado")
args = parser.parse_args()

cluster_id_to_train = args.cluster_id

# Configuracoes gerais
WEEKS_LOOKBACK = 48
WEEKS_FORECAST = 4

# Cria pastas
os.makedirs("outputs/models/cluster", exist_ok=True)
os.makedirs("outputs/metrics/cluster", exist_ok=True)
os.makedirs("outputs/history/cluster", exist_ok=True)
os.makedirs("outputs/plots/preds_vs_true", exist_ok=True)
os.makedirs("outputs/plots/losses", exist_ok=True)

# Extrai base se necessario
if not os.path.exists("df_base.csv"):
    with zipfile.ZipFile("df_base.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# Carrega base e clusters
df_base = pd.read_csv("df_base.csv")
df_clusters = pd.read_csv("top_clusters_salvos/agglo-agglo-geo3-time2-pca3_clusters.csv")
df_clusters["CD_MUN"] = df_clusters["CD_MUN"].astype(str)
df_base["CD_MUN"] = df_base["CD_MUN"].astype(str)
df_base = df_base.merge(df_clusters, on="CD_MUN", how="inner")

# Normalizacao
df_base["target_norm"] = df_base.groupby("CD_MUN")["target"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

# Funcoes auxiliares
def create_supervised_dataset(df, series_col="target_norm", lookback=48, forecast=4):
    df = df.sort_values("week")
    X, y = [], []
    for i in range(len(df) - lookback - forecast + 1):
        window = df[series_col].iloc[i:i + lookback].values
        label = df[series_col].iloc[i + lookback:i + lookback + forecast].sum()
        if not np.isnan(window).any() and not np.isnan(label):
            X.append(window)
            y.append(label)
    return np.array(X), np.array(y)

def prepare_cluster_data(df_cluster):
    all_X, all_y = [], []
    for _, df_mun in df_cluster.groupby("CD_MUN"):
        if df_mun["target_norm"].isna().sum() > 0:
            continue
        X, y = create_supervised_dataset(df_mun)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
    return np.concatenate(all_X), np.concatenate(all_y)

class DeepGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.batchnorm(out)
        out = self.relu(out)
        return self.output(out).squeeze()

# Treina GRU
print(f"\n🔁 Treinando GRU para cluster {cluster_id_to_train}...")
df_cluster = df_base[df_base["cluster_final"] == cluster_id_to_train]
X, y = prepare_cluster_data(df_cluster)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split train/val
dataset = TensorDataset(X_tensor, y_tensor)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

# Modelo
model = DeepGRU(input_size=1)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

# Treinamento
epochs = 250
patience = 20
best_loss = float('inf')
counter = 0
for epoch in trange(epochs, desc=f"Treinando GRU {cluster_id_to_train}"):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    model.eval()
    val_loss = np.mean([criterion(model(xb), yb).item() for xb, yb in val_loader])
    scheduler.step(val_loss)
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("⏹️ Early stopping!")
            break

model.load_state_dict(best_model_state)
torch.save(model.state_dict(), f"outputs/models/cluster/gru_cluster_{cluster_id_to_train}.pt")

# Treina XGBoost
print(f"\n🔁 Treinando XGBoost para cluster {cluster_id_to_train}...")
X_flat = X.reshape(X.shape[0], -1)
X_train, X_val, y_train, y_val = train_test_split(X_flat, y, test_size=0.2, random_state=42)
from xgboost import XGBRegressor
xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, objective="reg:squarederror")
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, f"outputs/models/cluster/xgb_cluster_{cluster_id_to_train}.joblib")

print(f"✅ Cluster {cluster_id_to_train} finalizado.")
