# 🧠 Machine Learning On-Chain
# 🧠 Machine Learning On-Chain

Repository with Unsupervised Machine Learning models applied to Ethereum wallet behavior.
Repositório com modelos de Machine Learning Não Supervisionado aplicados ao comportamento de carteiras Ethereum.

🇧🇷 [Leia em Português](#-português) | 🇬🇧 [Read in English](#-english)

## 🇧🇷 PORTUGUÊS

### 🎯 Objetivo
Aplicar técnicas de Machine Learning para segmentar perfis de carteiras (Clustering) e identificar comportamentos suspeitos (Detecção de Anomalias) em dados de blockchain.

### 🛠️ Tecnologias Usadas
*   **Python 3.10+**
*   **Pandas** (Manipulação de dados)
*   **Scikit-Learn** (Algoritmos K-Means e Isolation Forest)
*   **Matplotlib & Seaborn** (Visualização de dados)

### 📁 Estrutura
machine-learning-onchain/
├── ml_wallets.py # K-Means Clustering script
├── ml_anomalias.py # Isolation Forest script
├── kmeans_clusters.png # Clustering graph
├── anomaly_detection.png # Anomaly graph
├──README.md
└──READMEpt.md


### 📊 Projeto 1: Segmentação de Carteiras (K-Means)
*   **Como funciona:** O modelo analisou a contagem de transações, valor médio (ETH) e dias ativos de 700 carteiras simuladas. Os dados foram normalizados com `StandardScaler` e agrupados em 3 clusters com `KMeans`.
*   **Insights encontrados:**
    1.  **Baleias:** Poucas transações (~14), valor médio altíssimo (~479 ETH).
    2.  **Bots/Traders:** Muitas transações (~807), valor médio (~5 ETH).
    3.  **Varejo:** Transações normais (~150), valor baixo (~0.5 ETH).

### 🚨 Projeto 2: Detecção de Anomalias (Isolation Forest)
*   **Como funciona:** O modelo foi treinado para encontrar outliers no comportamento das carteiras. Injetei uma carteira "hacker" simulada (10.000 transações, média de 900 ETH em 1 dia).
*   **Insights encontrados:** O algoritmo `IsolationForest` sinalizou com sucesso a carteira hacker, provando sua eficiência em detectar possíveis fraudes, wash trading ou explorações de flash loans.

