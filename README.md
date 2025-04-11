# Previsão de Séries Temporais de Saúde Pública com Deep Learning por Município e Cluster

Este projeto realiza a previsão de **morbidade e mortalidade por doenças respiratórias e circulatórias** em municípios brasileiros, utilizando redes neurais recorrentes (GRU) e XGBoost. Os modelos são treinados **por município** e **por cluster de municípios**, com suporte total para execução em **supercomputador com gerenciador de filas PBS Pro**.

---

## Estrutura do Repositório

```bash
├── main_hpc.py              # Script principal (versão compatível com HPC)
├── job_treinamento.pbs      # Script de submissão PBS Pro
├── df_base.zip              # Base de dados principal (CSV comprimido)
├── top_clusters_salvos/     # Rótulos dos clusters por município
│   └── agglo-agglo-geo3-time2-pca3_clusters.csv
├── outputs/                 # Diretório gerado automaticamente com os resultados
│   ├── models/
│   │   ├── cluster/         # Modelos GRU e XGBoost por cluster
│   │   └── municipio/       # Modelos GRU por município
│   ├── metrics/             # Métricas salvas em JSON
│   ├── history/             # Histórico de loss por época
│   └── plots/               # Gráficos de avaliação
└── logs/
    ├── job_treinamento.log  # Log do PBS
    └── main_output.log      # Log do script principal
```

---

## 🚀 Execução no Supercomputador (PBS Pro)

1. **Crie seu ambiente Conda** (ajuste conforme seu cluster):

   ```bash
   conda create -n preditor_dl python=3.10
   conda activate preditor_dl
   pip install -r requirements.txt  
   ```

2. **Submeta o job no cluster:**

   ```bash
   qsub job_treinamento.pbs
   ```

3. **Monitore o status do job:**

   ```bash
   qstat -u $USER
   tail -f logs/main_output.log
   ```

---

## Modelos Utilizados

- **GRU (Gated Recurrent Units)**:
  - Treinados por cluster e por município
  - Janela de entrada: 48 semanas
  - Previsão: soma das 4 semanas seguintes

- **XGBoost**:
  - Treinado apenas por cluster
  - Utiliza as janelas como features tabulares

---

## 📈 Avaliação dos Modelos

Para cada modelo treinado, são salvos:

- Métricas de desempenho: MAE, RMSE, R²
- Gráficos de dispersão real vs predito
- Histograma e boxplot dos erros
- Histórico de loss por época

---

## Requisitos

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
torch
tqdm
joblib
```

---

##  Contato



 **[Pedro]** — pedro.cmg.camargo@unesp.br

