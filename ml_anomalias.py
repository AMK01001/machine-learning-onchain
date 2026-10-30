import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. GERAR DADOS SIMULADOS (O comportamento "Normal" da rede)
# ============================================================
np.random.seed(42) 

# Varejo
varejo = pd.DataFrame({
    'total_transactions': np.random.normal(150, 30, 500).astype(int),
    'avg_tx_value_eth': np.random.normal(0.5, 0.2, 500).clip(0.01),
    'active_days': np.random.normal(25, 3, 500).astype(int)
})

# Baleias
baleias = pd.DataFrame({
    'total_transactions': np.random.normal(15, 5, 50).astype(int),
    'avg_tx_value_eth': np.random.normal(500, 100, 50).clip(10),
    'active_days': np.random.normal(5, 2, 50).astype(int)
})

# Bots/Traders
bots = pd.DataFrame({
    'total_transactions': np.random.normal(800, 100, 150).astype(int),
    'avg_tx_value_eth': np.random.normal(5, 2, 150).clip(0.1),
    'active_days': np.random.normal(30, 1, 150).astype(int)
})

df_normal = pd.concat([varejo, baleias, bots], ignore_index=True)


# ============================================================
# 2. INJETANDO A ANOMALIA (O "Hacker" ou Wash Trader)
# ============================================================
# Vamos criar uma carteira bizarra: 10.000 transações, 900 ETH de média, tudo em 1 dia.
anomalia_hacker = pd.DataFrame({
    'total_transactions': [10000],
    'avg_tx_value_eth': [900.0],
    'active_days': [1]
})

# Juntando os dados normais com a anomalia
df = pd.concat([df_normal, anomalia_hacker], ignore_index=True)
features = ['total_transactions', 'avg_tx_value_eth', 'active_days']


# ============================================================
# 3. PREPARAÇÃO E TREINAMENTO DO ISOLATION FOREST
# ============================================================
# Normalizar os dados (importante para o Isolation Forest também)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Criar o modelo. O 'contamination=0.01' diz que esperamos ~1% de anomalias nos dados.
iso = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = iso.fit_predict(X_scaled)

# ============================================================
# 4. RESULTADOS
# ============================================================
print("--- Detecção de Anomalias ---")
anomalias = df[df['anomaly'] == -1] # -1 é a classe que o modelo usa para anomalias
print(f"Total de carteiras analisadas: {len(df)}")
print(f"Anomalias encontradas: {len(anomalias)}")
print("\nDetalhes da(s) anomalia(s):")
print(anomalias[features])

# ============================================================
# 5. VISUALIZAÇÃO GRÁFICA
# ============================================================

plt.figure(figsize=(10, 6))

# Criando uma coluna temporária apenas para o gráfico ficar correto visualmente
df['status'] = df['anomaly'].map({1: 'Normal', -1: 'Anomalia (Suspeito)'})

sns.scatterplot(
    data=df, 
    x='total_transactions', 
    y='avg_tx_value_eth', 
    hue='status', 
    palette={'Normal': '#377eb8', 'Anomalia (Suspeito)': '#e41a1c'}, 
    alpha=0.8,
    s=70
)
plt.title('Detecção de Anomalias com Isolation Forest')
plt.xlabel('Número de Transações')
plt.ylabel('Valor Médio da Transação (ETH)')
plt.yscale('log')
plt.show()


#Detecção de Anomalias (Isolation Forest)

# O que foi feito: O algoritmo foi treinado para encontrar "pontos fora da curva" (outliers), identificando comportamentos que 
# fogem do padrão normal da rede.

# O Resultado: O modelo detectou com sucesso uma carteira com comportamento de fraude/wash trading (10.000 transações em um 
# único dia movimentando 900 ETH). Além disso, identificou outras carteiras atípicas (baleias que movimentaram grandes volumes 
# em pouquíssimos dias).

