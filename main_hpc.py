#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn

# Parâmetros gerais
WEEKS_LOOKBACK = 48  # 12 meses
WEEKS_FORECAST = 4   # Acúmulo das próximas 4 semanas
    
# Diretório para salvar os modelos
os.makedirs("models_dl", exist_ok=True)
import zipfile

if not os.path.exists("df_base.csv"):
    with zipfile.ZipFile("df_base.zip", "r") as zip_ref:
        zip_ref.extractall(".")

df_base = pd.read_csv("df_base.csv")


# In[ ]:


import os

# Diretórios base
os.makedirs("outputs/models/cluster", exist_ok=True)
os.makedirs("outputs/models/municipio", exist_ok=True)
os.makedirs("outputs/metrics/cluster", exist_ok=True)
os.makedirs("outputs/metrics/municipio", exist_ok=True)
os.makedirs("outputs/history/cluster", exist_ok=True)
os.makedirs("outputs/history/municipio", exist_ok=True)
os.makedirs("outputs/plots/preds_vs_true", exist_ok=True)
os.makedirs("outputs/plots/losses", exist_ok=True)


# In[ ]:


# Carrega os rótulos de cluster salvos do melhor cenário
df_clusters = pd.read_csv("top_clusters_salvos/agglo-agglo-geo3-time2-pca3_clusters.csv")

# Garante que o código do município está no formato correto (caso necessário)
df_clusters["CD_MUN"] = df_clusters["CD_MUN"].astype(str)
df_base["CD_MUN"] = df_base["CD_MUN"].astype(str)

# Faz o merge com o df_base para adicionar o cluster_final
df_base = df_base.merge(df_clusters, on="CD_MUN", how="inner")

# Verifica a distribuição de municípios por cluster
print(\1))


# In[ ]:


# Normaliza o target por município (z-score)
df_base["target_norm"] = df_base.groupby("CD_MUN")["target"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)


# In[ ]:


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

def prepare_cluster_data(df_cluster, lookback=48, forecast=4, series_col="target_norm"):
    all_X, all_y = [], []
    for _, df_mun in df_cluster.groupby("CD_MUN"):
        if df_mun[series_col].isna().sum() > 0:
            continue
        X, y = create_supervised_dataset(df_mun, series_col, lookback, forecast)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
    return np.concatenate(all_X), np.concatenate(all_y)


# In[ ]:


class DeepGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # última saída da sequência
        out = self.batchnorm(out)
        out = self.relu(out)
        return self.output(out).squeeze()


# In[ ]:


from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

def get_dataloaders_from_cluster(df, cluster_id, lookback=48, forecast=4, batch_size=128, val_split=0.2):
    df_cluster = df[df["cluster_final"] == cluster_id]
    X, y = prepare_cluster_data(df_cluster, lookback, forecast)

    # Transforma para tensores PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [batch, seq_len, 1]
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# In[ ]:


from tqdm import trange

def train_model(model, train_loader, val_loader, n_epochs=250, lr=1e-3, patience=20):
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=8, verbose=False)

    best_loss = float("inf")
    counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    epoch_bar = trange(n_epochs, desc="Treinando modelo", leave=True)

    for epoch in epoch_bar:
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss = criterion(preds, yb).item()
                val_losses.append(val_loss)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Atualiza tqdm com as métricas da época
        epoch_bar.set_description(f"📉 Epoch {epoch+1}/{n_epochs}")
        epoch_bar.set_postfix({
            "Train": f"{avg_train_loss:.4f}",
            "Val": f"{avg_val_loss:.4f}",
            "Patience": f"{counter}/{patience}"
        })

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping!")
                break

    model.load_state_dict(best_state)
    return model, history


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def plot_all_eval_metrics(model, loader, cluster_id):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb).cpu().numpy())
            targets.extend(yb.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)
    errors = preds - targets

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Curva real vs predição (dispersão)
    axs[0, 0].scatter(targets, preds, alpha=0.5)
    axs[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    axs[0, 0].set_title(f" Dispersão - Cluster {cluster_id}")
    axs[0, 0].set_xlabel("Valor real")
    axs[0, 0].set_ylabel("Predição")
    axs[0, 0].grid(True)

    # 2. Boxplot dos erros
    sns.boxplot(x=errors, ax=axs[0, 1], color="skyblue")
    axs[0, 1].set_title(f" Boxplot de erros - Cluster {cluster_id}")
    axs[0, 1].set_xlabel("Erro")
    axs[0, 1].grid(True)

    # 3. Histograma dos erros
    axs[1, 0].hist(errors, bins=30, edgecolor="black", color="lightcoral")
    axs[1, 0].set_title(f" Histograma de erros - Cluster {cluster_id}")
    axs[1, 0].set_xlabel("Erro")
    axs[1, 0].set_ylabel("Frequência")
    axs[1, 0].grid(True)

    # 4. Erro absoluto ao longo do tempo (opcional, pode tirar)
    axs[1, 1].plot(np.abs(errors), alpha=0.7)
    axs[1, 1].set_title(f" Erro absoluto por amostra - Cluster {cluster_id}")
    axs[1, 1].set_xlabel("Amostra")
    axs[1, 1].set_ylabel("Erro Absoluto")
    axs[1, 1].grid(True)

    plt.suptitle(f" Avaliação - Cluster {cluster_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    


# In[ ]:


models_by_cluster = {}

for cluster_id in sorted(df_base["cluster_final"].unique()):
    print(f"\n🔁 Treinando cluster {cluster_id}...")
    train_loader, val_loader = get_dataloaders_from_cluster(df_base, cluster_id)

    model = DeepGRU(input_size=1)
    model, history = train_model(model, train_loader, val_loader)

    # Salva o modelo
    torch.save(model.state_dict(), f"outputs/models/cluster/gru_cluster_{cluster_id}.pt")

    models_by_cluster[cluster_id] = model



# In[ ]:


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

os.makedirs("models_xgb", exist_ok=True)

xgb_metrics = {}

for cluster_id in sorted(df_base["cluster_final"].unique()):
    print(f"\nTreinando XGBoost para cluster {cluster_id}...")

    df_cluster = df_base[df_base["cluster_final"] == cluster_id].copy()

    # Agrupa e cria janela de contexto como features
    X_all, y_all = prepare_cluster_data(df_cluster, lookback=WEEKS_LOOKBACK, forecast=WEEKS_FORECAST)

    # Ajusta formato para XGBoost
    X_flat = X_all.reshape(X_all.shape[0], -1)

    X_train, X_val, y_train, y_val = train_test_split(X_flat, y_all, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, objective="reg:squarederror")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    print(f"✅ RMSE: {rmse:.4f}")
    xgb_metrics[cluster_id] = rmse

    joblib.dump(model, f"outputs/models/cluster/xgb_cluster_{cluster_id}.joblib")



# In[ ]:


import time
import torch.optim as optim


models_by_municipio = {}

unique_muns = df_base["CD_MUN"].unique()

for i, mun in enumerate(unique_muns):
    df_mun = df_base[df_base["CD_MUN"] == mun]

    if df_mun["target_norm"].isna().sum() > 0:
        continue

    X, y = create_supervised_dataset(df_mun, series_col="target_norm", lookback=WEEKS_LOOKBACK, forecast=WEEKS_FORECAST)

    if len(X) < 10:
        continue  # pula séries com pouca informação

    # Split simples
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Prepara os loaders
    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train).float().unsqueeze(-1),
                                              torch.tensor(y_train).float())
    val_ds = torch.utils.data.TensorDataset(torch.tensor(X_val).float().unsqueeze(-1),
                                            torch.tensor(y_val).float())

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)

    # Modelo GRU
    model = DeepGRU(input_size=1)

    print(f"\n🧠 Treinando município {mun} ({i+1}/{len(unique_muns)})")
    start = time.time()
    model, history = train_model(model, train_loader, val_loader)
    duration = time.time() - start
    print(f"⏱️ Tempo de treinamento: {duration:.1f} segundos")

    # Salva modelo
    torch.save(model.state_dict(), f"outputs/models/municipio/gru_mun_{mun}.pt")

    models_by_municipio[mun] = model


# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model(model, loader):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
            all_preds.extend(preds.numpy())
            all_true.extend(yb.numpy())
    return np.array(all_preds), np.array(all_true)

def plot_preds_vs_true(y_true, y_pred, title=""):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Real")
    plt.ylabel("Predito")
    plt.title(f"🔍 Real vs Predito {title}")
    plt.grid(True)
    plt.tight_layout()
    

def print_metrics(y_true, y_pred):
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):.4f}")
    print(f"R²: {r2_score(y_true, y_pred):.4f}")


# In[ ]:


for cluster_id in sorted(df_base["cluster_final"].unique()):
    print(f"\n📊 Avaliando modelo GRU do cluster {cluster_id}")

    train_loader, val_loader = get_dataloaders_from_cluster(df_base, cluster_id)
    model = DeepGRU(input_size=1)
    model.load_state_dict(torch.load(f"models_dl/gru_cluster_{cluster_id}.pt"))
    
    preds, true = evaluate_model(model, val_loader)

    print_metrics(true, preds)
    plot_preds_vs_true(true, preds, title=f"Cluster {cluster_id}")


# In[ ]:


metricas_cluster = []

for cluster_id in sorted(df_base["cluster_final"].unique()):
    model = DeepGRU(input_size=1)
    model.load_state_dict(torch.load(f"models_dl/gru_cluster_{cluster_id}.pt"))
    _, val_loader = get_dataloaders_from_cluster(df_base, cluster_id)

    preds, true = evaluate_model(model, val_loader)
    mae = mean_absolute_error(true, preds)
    rmse = mean_squared_error(true, preds, squared=False)
    r2 = r2_score(true, preds)

    metricas_cluster.append({
        "cluster": cluster_id,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

df_metricas = pd.DataFrame(metricas_cluster)
print(\1))


# In[ ]:


import json

def salvar_metricas_e_avaliacoes(model, loader, modelo_id, tipo="cluster"):
    y_pred, y_true = evaluate_model(model, loader)

    # Calcula métricas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    metricas = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    # Salva em JSON
    path_json = f"outputs/metrics/{tipo}/{modelo_id}.json"
    with open(path_json, "w") as f:
        json.dump(metricas, f, indent=2)

    # Salva histórico, se existir
    if modelo_id in history_by_cluster:
        path_hist = f"outputs/history/{tipo}/{modelo_id}.json"
        with open(path_hist, "w") as f:
            json.dump(history_by_cluster[modelo_id], f, indent=2)

    # Salva gráficos
    plt.figure()
    plot_preds_vs_true(y_true, y_pred, title=f"{tipo.capitalize()} {modelo_id}")
    plt.savefig(f"outputs/plots/preds_vs_true/{tipo}_{modelo_id}.png")
    plt.close()

    plot_all_eval_metrics(model, loader, modelo_id)
    plt.savefig(f"outputs/plots/losses/{tipo}_{modelo_id}.png")
    plt.close()

    print(f"✅ Avaliação salva para {tipo} {modelo_id}")

# Avaliação dos modelos por cluster
for cluster_id in models_by_cluster.keys():
    print(f"\n📊 Avaliando modelo do cluster {cluster_id}")
    train_loader, val_loader = get_dataloaders_from_cluster(df_base, cluster_id)
    model = models_by_cluster[cluster_id]
    salvar_metricas_e_avaliacoes(model, val_loader, cluster_id, tipo="cluster")

# Avaliação dos modelos por município
for mun_id in models_by_municipio.keys():
    print(f"\n📊 Avaliando modelo do município {mun_id}")
    df_mun = df_base[df_base["CD_MUN"] == mun_id]
    X, y = create_supervised_dataset(df_mun, series_col="target_norm")
    if len(X) == 0: continue

    split = int(0.8 * len(X))
    X_val, y_val = X[split:], y[split:]

    val_ds = torch.utils.data.TensorDataset(torch.tensor(X_val).float().unsqueeze(-1),
                                            torch.tensor(y_val).float())
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)

    model = models_by_municipio[mun_id]
    salvar_metricas_e_avaliacoes(model, val_loader, mun_id, tipo="municipio")

