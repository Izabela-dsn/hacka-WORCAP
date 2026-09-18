import os
import glob
import gc
import numpy as np
import pandas as pd
import xarray as xr

print("=== PIPELINE COM TODAS AS VARIÁVEIS ATMOSFÉRICAS DA ERA5 ===")

DATA_DIR = '/content/'
DATA_INICIO = '2000-01-01'  # Período selecionado para manter o uso de RAM seguro

path_tp = os.path.join(DATA_DIR, "treino_tp.nc")
path_alvo = os.path.join(DATA_DIR, "treino_tp_alvo.nc")

# 1. Identificar e mesclar TODOS os arquivos treino_*.nc presentes no diretório
print(f"\n[1/5] Carregando e unificando todos os arquivos NetCDF a partir de {DATA_INICIO}...")
ds_tp = xr.open_dataset(path_tp).sel(time=slice(DATA_INICIO, None))
ds_alvo = xr.open_dataset(path_alvo).sel(time=slice(DATA_INICIO, None))

# Encontrar automaticamente todos os outros arquivos de variáveis atmosféricas
outros_arquivos = sorted([
    f for f in glob.glob(os.path.join(DATA_DIR, "treino_*.nc"))
    if f not in [path_tp, path_alvo]
])

datasets = [ds_tp, ds_alvo]
for arq in outros_arquivos:
    nome_var = os.path.basename(arq)
    print(f" -> Incluindo variável física: {nome_var}")
    datasets.append(xr.open_dataset(arq).sel(time=slice(DATA_INICIO, None)))

ds_treino = xr.merge(datasets)
del datasets, ds_tp, ds_alvo
gc.collect()

# 2. Criar Lags da precipitação e variáveis temporais
print("\n[2/5] Criando features de ciclo sazonal e lags...")
meses = ds_treino['time.month']
ds_treino['sin_mes'] = np.sin(2 * np.pi * meses / 12).astype(np.float32)
ds_treino['cos_mes'] = np.cos(2 * np.pi * meses / 12).astype(np.float32)

ds_treino['tp_lag1'] = ds_treino['tp'].shift(time=1).astype(np.float32)
ds_treino['tp_lag2'] = ds_treino['tp'].shift(time=2).astype(np.float32)

# Mapear TODAS as variáveis físicas carregadas
variaveis_iniciais = ['tp', 'tp_lag1', 'tp_lag2', 'sin_mes', 'cos_mes']
outras_vars = [v for v in ds_treino.data_vars if v not in ['tp_alvo'] + variaveis_iniciais]
features = variaveis_iniciais + outras_vars

print(f"\nVariáveis físicas integradas no modelo ({len(features)} no total):")
for f_name in features:
    print(f"  • {f_name}")

# 3. Converter a grade 3D para tabela 2D (DataFrame)
print("\n[3/5] Convertendo grade 3D em tabela 2D...")
df = ds_treino[features + ['tp_alvo']].to_dataframe().reset_index()

del ds_treino
gc.collect()

for col in features:
    df[col] = df[col].astype(np.float32)
df['tp_alvo'] = df['tp_alvo'].astype(np.float32)
df['mes_alvo'] = (df['time'].dt.month % 12) + 1

df = df.dropna(subset=['tp_alvo'] + features)
gc.collect()

# 4. Calcular a Climatologia Leve no xarray (< 2015)
print("\n[4/5] Calculando Climatologia Histórica de precipitação no xarray...")
ds_tp_treino = xr.open_dataset(path_tp).sel(time=slice(DATA_INICIO, '2014-12-31'))
clim_xr = ds_tp_treino['tp'].groupby('time.month').mean(dim='time').rename('clim_alvo')

clim_df = clim_xr.to_dataframe().reset_index()
clim_df.rename(columns={'month': 'mes_alvo'}, inplace=True)
clim_df['clim_alvo'] = clim_df['clim_alvo'].astype(np.float32)

del ds_tp_treino, clim_xr
gc.collect()

# 5. Divisão Treino/Validação e Alvo de Anomalia
print("\n[5/5] Dividindo conjuntos e mesclando com a Climatologia...")
df_train = df[df['time'].dt.year < 2015].copy()
df_val = df[df['time'].dt.year >= 2015].copy()
del df
gc.collect()

df_train = df_train.merge(clim_df, on=['mes_alvo', 'lat', 'lon'], how='left')
df_val = df_val.merge(clim_df, on=['mes_alvo', 'lat', 'lon'], how='left')

# Calcular a Anomalia Alvo (Chuva Real - Climatologia Local)
df_train['anomalia_alvo'] = df_train['tp_alvo'] - df_train['clim_alvo']
df_val['anomalia_alvo'] = df_val['tp_alvo'] - df_val['clim_alvo']

features_modelo = features + ['lat', 'lon', 'clim_alvo']

X_train, y_train = df_train[features_modelo], df_train['anomalia_alvo']
X_val, y_val = df_val[features_modelo], df_val['anomalia_alvo']

print("\n" + "="*60)
print("[SUCESSO] DADOS PRONTOS COM TODAS AS VARIÁVEIS ATMOSFÉRICAS!")
print("="*60)
print(f"• Total de Features de Entrada : {len(features_modelo)}")
print(f"• Amostras de Treino            : {len(X_train):,}")
print(f"• Amostras de Validação         : {len(X_val):,}")
