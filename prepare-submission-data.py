import os
import numpy as np
import pandas as pd
import xarray as xr
import joblib
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

print("=== INFERÊNCIA COM TODAS AS VARIÁVEIS ATMOSFÉRICAS (2023-2024) ===")

DATA_DIR = '/content/'

model = joblib.load("modelo_lightgbm.pkl")
features_modelo = joblib.load("features_list.pkl")
clim_df = joblib.load("climatologia_df.pkl")

path_teste = os.path.join(DATA_DIR, "teste_features.nc")
print(f"Carregando arquivo de teste: '{path_teste}'...")
ds_teste = xr.open_dataset(path_teste)

if 'time' in ds_teste.coords:
    meses = ds_teste['time.month']
    ds_teste['sin_mes'] = np.sin(2 * np.pi * meses / 12).astype(np.float32)
    ds_teste['cos_mes'] = np.cos(2 * np.pi * meses / 12).astype(np.float32)

df_teste = ds_teste.to_dataframe().reset_index()
df_teste['mes_alvo'] = df_teste['time'].dt.month

# Mesclar climatologia por (mes_alvo, lat, lon)
df_teste = df_teste.merge(clim_df, on=['mes_alvo', 'lat', 'lon'], how='left')
df_teste['clim_alvo'] = df_teste['clim_alvo'].fillna(df_teste['clim_alvo'].mean())

# Garantir que todas as colunas usadas no treino estejam presentes
for col in features_modelo:
    if col not in df_teste.columns:
        df_teste[col] = 0.0

X_test = df_teste[features_modelo]

# Predição da anomalia e reconstrução da chuva
print(f"Gerando previsões para {len(X_test):,} pontos de grade...")
pred_anomalia_test = model.predict(X_test)
pred_final = np.clip(df_teste['clim_alvo'] + pred_anomalia_test, 0, None)

path_sample = os.path.join(DATA_DIR, "sample_submission.csv")
sub = pd.read_csv(path_sample)
sub['tp_mm_day'] = pred_final.values

output_csv = "submission_lightgbm_multivariado2.csv"
sub.to_csv(output_csv, index=False)

print(f"\n[SUCESSO] Arquivo gerado: '{output_csv}'!")
files.download(output_csv)
