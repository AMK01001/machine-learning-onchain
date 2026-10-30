# 🧠 Machine Learning On-Chain
# 🧠 Machine Learning On-Chain

Repository with Unsupervised Machine Learning models applied to Ethereum wallet behavior.
Repositório com modelos de Machine Learning Não Supervisionado aplicados ao comportamento de carteiras Ethereum.

🇧🇷 [Leia em Português](https://github.com/AMK01001/machine-learning-onchain/blob/main/READMEpt.md) | 
🇬🇧 [Read in English](https://github.com/AMK01001/machine-learning-onchain/blob/main/README.md)

---

## 🇬🇧 ENGLISH

### 🎯 Objective
Apply Machine Learning techniques to segment wallet profiles (Clustering) and identify suspicious behaviors (Anomaly Detection) in blockchain data.

### 🛠️ Technologies Used
*   **Python 3.10+**
*   **Pandas** (Data manipulation)
*   **Scikit-Learn** (K-Means and Isolation Forest algorithms)
*   **Matplotlib & Seaborn** (Data visualization)

### 📁 Structure
machine-learning-onchain/
├── ml_wallets.py # K-Means Clustering script
├── ml_anomalias.py # Isolation Forest script
├── kmeans_clusters.png # Clustering graph
├── anomaly_detection.png # Anomaly graph
├──README.md
└──READMEpt.md


### 📊 Project 1: Wallet Segmentation (K-Means)
*   **How it works:** The model analyzed transaction count, average transaction value (ETH), and active days for 700 simulated wallets. Data was normalized using `StandardScaler` and grouped into 3 clusters using `KMeans`.
*   **Insights found:**
    1.  **Whales:** Few transactions (~14), extremely high average value (~479 ETH).
    2.  **Bots/Traders:** High transaction count (~807), medium average value (~5 ETH).
    3.  **Retail:** Normal transaction count (~150), low average value (~0.5 ETH).

### 🚨 Project 2: Anomaly Detection (Isolation Forest)
*   **How it works:** The model was trained to find outliers in wallet behavior. I injected a simulated "hacker" wallet (10,000 transactions, 900 ETH average in 1 day).
*   **Insights found:** The `IsolationForest` algorithm successfully flagged the hacker wallet, proving its efficiency in detecting potential fraud, wash trading, or flash loan exploits.





