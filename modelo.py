import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Substitui pelo link de exportação CSV da tua planilha do Google Sheets
csv_url = 'https://docs.google.com/spreadsheets/d/1GYHsALBmPJCcy5xU7uhUsnZgfHjYguZmaqGeLAnbYUQ/export?format=csv'

print("1. A carregar os dados diretamente do Google Sheets...")
df = pd.read_csv(csv_url)

# 2. Seleção das colunas numéricas (Ajusta os nomes das colunas conforme a tua planilha)
# Certifica-te de que estas colunas existem no teu Google Sheets
features = ['danceability', 'energy', 'valence', 'tempo'] 
X = df[features].dropna()

print("2. A normalizar os dados (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("3. A treinar o modelo K-Means...")
# K=4 ou K=5 dependendo de quantos agrupamentos pretendes
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 4. Guardar os artefactos gerados (model.pkl e scaler.pkl)
pickle.dump(kmeans, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Modelo treinado e ficheiros 'model.pkl' e 'scaler.pkl' exportados com sucesso!")