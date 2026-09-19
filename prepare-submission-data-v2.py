import os
import numpy as np
import pandas as pd
import xarray as xr
import joblib
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

print("=== INFERÊNCIA COM 12 MODELOS MENSAIS (2023-2024) ===")

DATA_DIR = '/content/'

# 1. Carregar os 12 Modelos, Lista de Features e Climatologia
modelos_mensais = joblib.load("modelos_mensais.pkl")
features_modelo_exp = joblib.load("features_list.pkl")
clim_df = joblib.load("climatologia_df.pkl")

# 2. Carregar e preparar o teste
path_teste = os.path.join(DATA_DIR, "teste_features.nc")
print(f"Carregando arquivo de teste: '{path_teste}'...")
ds_teste = xr.open_dataset(path_teste)

if 'time' in ds_teste.coords:
    meses = ds_teste['time.month']
    ds_teste['sin_mes'] = np.sin(2 * np.pi * meses / 12).astype(np.float32)
    ds_teste['cos_mes'] = np.cos(2 * np.pi * meses / 12).astype(np.float32)

df_teste = ds_teste.to_dataframe().reset_index()
df_teste['mes_alvo'] = df_teste['time'].dt.month

# Criar fluxo de umidade no teste
col_umidade = [c for c in df_teste.columns if 'shum' in c or 'q_850' in c or 'rel' in c or 'r_850' in c]
if col_umidade and 'u_850' in df_teste.columns and 'v_850' in df_teste.columns:
    df_teste['fluxo_u'] = (df_teste['u_850'] * df_teste[col_umidade[0]]).astype(np.float32)
    df_teste['fluxo_v'] = (df_teste['v_850'] * df_teste[col_umidade[0]]).astype(np.float32)

# Mesclar climatologia por (mes_alvo, lat, lon)
df_teste = df_teste.merge(clim_df, on=['mes_alvo', 'lat', 'lon'], how='left')
df_teste['clim_alvo'] = df_teste['clim_alvo'].fillna(df_teste['clim_alvo'].mean())

for col in features_modelo_exp:
    if col not in df_teste.columns:
        df_teste[col] = 0.0

# 3. Predição da Anomalia mês a mês usando seu modelo especializado
print(f"Gerando previsões especializadas para os 24 meses do teste...")
pred_anomalia_test = np.zeros(len(df_teste))

for m in range(1, 13):
    idx_m = df_teste['mes_alvo'] == m
    if idx_m.sum() > 0:
        X_test_m = df_teste.loc[idx_m, features_modelo_exp]
        pred_anomalia_test[idx_m] = modelos_mensais[m].predict(X_test_m)

# Reconstruir a chuva total (Climatologia + Anomalia Prevista)
pred_final = np.clip(df_teste['clim_alvo'] + pred_anomalia_test, 0, None)

# 4. Gerar CSV final
path_sample = os.path.join(DATA_DIR, "sample_submission.csv")
sub = pd.read_csv(path_sample)
sub['tp_mm_day'] = pred_final.values

output_csv = "submission_lightgbm_12modelos.csv"
sub.to_csv(output_csv, index=False)

print(f"\n[SUCESSO] Previsões salvas em '{output_csv}'!")
files.download(output_csv)
