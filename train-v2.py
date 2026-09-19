import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
import numpy as np
import joblib

print("=== TREINANDO 12 MODELOS ESPECIALIZADOS POR MÊS + FLUXO DE UMIDADE ===")

# 1. Criar variáveis de Fluxo de Umidade (se existirem umidade e ventos)
for dataset_df in [df_train, df_val]:
    col_umidade = [c for c in dataset_df.columns if 'shum' in c or 'q_850' in c or 'rel' in c or 'r_850' in c]
    if col_umidade and 'u_850' in dataset_df.columns and 'v_850' in dataset_df.columns:
        u_col = dataset_df['u_850']
        v_col = dataset_df['v_850']
        q_col = dataset_df[col_umidade[0]]
        dataset_df['fluxo_u'] = (u_col * q_col).astype(np.float32)
        dataset_df['fluxo_v'] = (v_col * q_col).astype(np.float32)

# Atualizar lista de features do modelo
features_modelo_exp = [c for c in df_train.columns if c not in ['time', 'tp_alvo', 'anomalia_alvo', 'mes_alvo']]

print(f"Total de {len(features_modelo_exp)} features configuradas.")

# 2. Treinar 12 modelos (um para cada mês do ano)
modelos_mensais = {}
pred_anomalia_val = np.zeros(len(df_val))

for m in range(1, 13):
    print(f" -> Treinando modelo especializado para o Mês {m:02d}...")

    # Filtrar apenas dados do mês m no treino e validação
    idx_tr = df_train['mes_alvo'] == m
    idx_va = df_val['mes_alvo'] == m

    X_tr_m, y_tr_m = df_train.loc[idx_tr, features_modelo_exp], df_train.loc[idx_tr, 'anomalia_alvo']
    X_va_m, y_va_m = df_val.loc[idx_va, features_modelo_exp], df_val.loc[idx_va, 'anomalia_alvo']

    model_m = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model_m.fit(
        X_tr_m, y_tr_m,
        eval_set=[(X_va_m, y_va_m)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    modelos_mensais[m] = model_m
    pred_anomalia_val[idx_va] = model_m.predict(X_va_m)

# 3. Avaliar chuva total reconstruída na Validação
pred_tp_val = np.clip(df_val['clim_alvo'] + pred_anomalia_val, 0, None)

rmse_baseline = root_mean_squared_error(df_val['tp_alvo'], df_val['clim_alvo'])
rmse_modelo = root_mean_squared_error(df_val['tp_alvo'], pred_tp_val)

print("\n" + "="*60)
print("     RESULTADO COM 12 MODELOS MENSAIS ESPECIALIZADOS")
print("="*60)
print(f"• Baseline Climatologia  : {rmse_baseline:.4f} mm/dia")
print(f"• RMSE 12 Modelos Mensais : {rmse_modelo:.4f} mm/dia")
print(f"• Ganho de Acurácia      : {(rmse_baseline - rmse_modelo):.4f} mm/dia")
print("="*60)

# Salvar dicionário de modelos
joblib.dump(modelos_mensais, "modelos_mensais.pkl")
joblib.dump(features_modelo_exp, "features_list.pkl")
joblib.dump(clim_df, "climatologia_df.pkl")
