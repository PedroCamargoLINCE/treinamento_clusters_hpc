# Previsão de Séries Temporais de Saúde Pública com XGBoost por Município e Cluster

Este projeto realiza a previsão de **morbidade e mortalidade por doenças respiratórias e circulatórias** em municípios brasileiros, utilizando **XGBoost**. Os modelos são treinados **por município** e/ou **por cluster de municípios**.

---

## Estrutura do Repositório

```bash
├── treinoxgboost.ipynb      # Script principal para treinamento com XGBoost (Jupyter Notebook)
├── df_base.zip              # Base de dados principal (CSV comprimido)
├── requirements.txt         # Dependências do projeto
├── top_clusters_salvos/     # Rótulos dos clusters por município (ex: agglo-agglo-geo3-time2-pca3_clusters.csv)
├── models/                  # Diretório com os modelos XGBoost treinados
│   └── xgboost_YYYYMMDD_HHMMSS/ # Modelos de uma execução específica (timestamped)
│       └── COD_MUNICIPIO/   # Ex: Modelos por município ou ID de cluster
│           └── model.xgb    # Exemplo de arquivo de modelo salvo
├── results/                 # Diretório com os resultados da avaliação dos modelos
│   └── xgboost_YYYYMMDD_HHMMSS/ # Resultados de uma execução específica (timestamped)
│       ├── metrics.json     # Exemplo: Métricas de desempenho salvas
│       └── plots/           # Exemplo: Gráficos de avaliação (dispersão, erros, etc.)
└── README.md                # Este arquivo
```

---

##  Execução Local com Jupyter Notebook

1.  **Crie seu ambiente Conda** (ajuste conforme necessário):

    ```bash
    conda create -n preditor_xgb python=3.10
    conda activate preditor_xgb
    pip install -r requirements.txt
    # Certifique-se de ter o Jupyter Notebook ou Jupyter Lab instalado
    # pip install notebook jupyterlab 
    ```

2.  **Descomprima a base de dados `df_base.zip`** se necessário, e coloque `df_base.csv` no local esperado pelo notebook (e.g., `data/df_base.csv` ou no diretório raiz).

3.  **Execute o Jupyter Notebook:**

    Abra o `treinoxgboost.ipynb` utilizando Jupyter Notebook ou Jupyter Lab e execute as células para treinar os modelos e gerar os resultados.

    ```bash
    jupyter lab # ou jupyter notebook
    ```

---

## Modelos Utilizados

-   **XGBoost**:
    -   Treinado por cluster e/ou por município.
    -   Utiliza janelas temporais como features tabulares.
    -   Janela de entrada: (Verificar no notebook, exemplo: 48 semanas)
    -   Previsão: (Verificar no notebook, exemplo: soma das 4 semanas seguintes)

---

##  Avaliação dos Modelos

Para cada modelo treinado (ou conjunto de modelos de uma execução), são salvos no diretório `results/xgboost_YYYYMMDD_HHMMSS/`:

-   **Métricas de desempenho**: MAE, RMSE, R² (geralmente em arquivos JSON).
-   **Gráficos de avaliação**:
    -   Dispersão real vs. predito.
    -   Histograma e boxplot dos erros.
-   Os modelos treinados são salvos em `models/xgboost_YYYYMMDD_HHMMSS/`.

---

## Requisitos

As dependências do projeto estão listadas no arquivo `requirements.txt`.

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
# torch (se ainda for usado para alguma etapa de pré-processamento ou comparativo)
tqdm
joblib
ipykernel*
```

---

## Contato

**[Pedro]** — pedro.cmg.camargo@unesp.br

