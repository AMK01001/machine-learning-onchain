import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. GERAR DADOS SIMULADOS (Como se tivéssemos baixado do Dune)
# ============================================================
np.random.seed(42) # Para os resultados serem iguais toda vez que rodar
num_wallets = 1000

# Criando 3 "personas" (perfis) de carteiras
# Perfil 0: Varejo (Muitas transações, valor baixo)
varejo = pd.DataFrame({
    'total_transactions': np.random.normal(150, 30, 500).astype(int),
    'avg_tx_value_eth': np.random.normal(0.5, 0.2, 500).clip(0.01),
    'active_days': np.random.normal(25, 3, 500).astype(int)
})

# Perfil 1: Baleias (Poucas transações, valor altíssimo)
baleias = pd.DataFrame({
    'total_transactions': np.random.normal(15, 5, 50).astype(int),
    'avg_tx_value_eth': np.random.normal(500, 100, 50).clip(10),
    'active_days': np.random.normal(5, 2, 50).astype(int)
})

# Perfil 2: Bots/Traders (Muitas transações, valor médio)
bots = pd.DataFrame({
    'total_transactions': np.random.normal(800, 100, 150).astype(int),
    'avg_tx_value_eth': np.random.normal(5, 2, 150).clip(0.1),
    'active_days': np.random.normal(30, 1, 150).astype(int)
})

# Juntando tudo no mesmo "banco de dados"
df = pd.concat([varejo, baleias, bots], ignore_index=True)

# ============================================================
# 2. PREPARAÇÃO DOS DADOS PARA O K-MEANS
# ============================================================
features = ['total_transactions', 'avg_tx_value_eth', 'active_days']
X = df[features]

# Normalização (Obrigatório para K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 3. TREINANDO O MODELO DE CLUSTERING
# ============================================================
# Dizemos ao algoritmo que queremos 3 grupos (n_clusters=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# ============================================================
# 4. ANALISANDO O QUE O MODELO DESCOBRIU
# ============================================================
print("--- Média das características por Cluster ---")
resultado = df.groupby('cluster')[features].mean().round(2)
print(resultado)

# Identificando quem é quem (baseado na lógica de negócios)
print("\n--- Interpretação dos Clusters ---")
for cluster_id in resultado.index:
    media_valor = resultado.loc[cluster_id, 'avg_tx_value_eth']
    media_tx = resultado.loc[cluster_id, 'total_transactions']
    
    if media_valor > 100:
        print(f"Cluster {cluster_id}: 🐋 BALEIAS (Poucas transações, valor altíssimo: {media_valor:.2f} ETH)")
    elif media_tx > 500:
        print(f"Cluster {cluster_id}: 🤖 BOTS/TRADERS (Muitas transações: {media_tx:.0f}, valor médio: {media_valor:.2f} ETH)")
    else:
        print(f"Cluster {cluster_id}: 🛒 VAREJO (Transações normais: {media_tx:.0f}, valor: {media_valor:.2f} ETH)")

# ============================================================
# 5. VISUALIZAÇÃO GRÁFICA (Scatter Plot)
# ============================================================
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, 
    x='total_transactions', 
    y='avg_tx_value_eth', 
    hue='cluster', 
    palette='viridis', 
    alpha=0.7
)
plt.title('Segmentação de Carteiras (K-Means)')
plt.xlabel('Número de Transações')
plt.ylabel('Valor Médio da Transação (ETH)')
# Deixar o eixo Y em escala logarítmica para vermos bem as baleias
plt.yscale('log') 
plt.show()


#  Analisando o seu Gráfico e o Resultado
#O gráfico de dispersão (Scatter Plot) é a prova visual do que o Machine Learning descobriu:

#🟡 Cluster 2 (Amarelo - Canto Superior Esquerdo): As Baleias 🐋

#O que o gráfico mostra: Pontinhos bem à esquerda (poucas transações, menos de 50) e lá no alto (valor altíssimo, perto de 1.000 ETH).

#O que o terminal disse: total_transactions: 14, avg_tx_value_eth: 479. São poucos movimentos, mas cada movimento movimenta uma fortuna.

#🟣 Cluster 0 (Roxo - Canto Inferior Esquerdo): O Varejo 🛒

#O que o gráfico mostra: Uma nuvem densa de pontos embaixo (valor baixo, menos de 1 ETH) e com um número moderado de transações
# (entre 50 e 250).

#O que o terminal disse: total_transactions: 150, avg_tx_value_eth: 0.51. É o usuário comum, que faz transações no dia a dia com 
# valores pequenos.

#🟢 Cluster 1 (Verde - Canto Direito): Os Bots/Traders 🤖

#O que o gráfico mostra: Pontos espalhados para a direita (mais de 500 transações) com um valor médio (entre 1 e 10 ETH).

#O que o terminal disse: total_transactions: 807, avg_tx_value_eth: 4.96. São carteiras que não param de trabalhar, 
# típicas de robôs de arbitragem ou traders de alta frequência.

