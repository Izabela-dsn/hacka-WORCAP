import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
import joblib

print("=== TREINANDO LIGHTGBM MULTIVARIADO (ANOMALIA) ===")

model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.015,
    num_leaves=31,
    max_depth=6,
    min_child_samples=100,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,   # Amostra 70% das variáveis atmosféricas por árvore
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50, verbose=True)
    ]
)

# Avaliação na Validação
print("\nAvaliando no bloco de validação (2015-2022)...")
pred_anomalia_val = model.predict(X_val)
pred_tp_val = np.clip(df_val['clim_alvo'] + pred_anomalia_val, 0, None)

rmse_baseline = root_mean_squared_error(df_val['tp_alvo'], df_val['clim_alvo'])
rmse_modelo = root_mean_squared_error(df_val['tp_alvo'], pred_tp_val)

print("\n" + "="*60)
print("     RESULTADOS DA VALIDAÇÃO MULTIVARIADA (2015-2022)")
print("="*60)
print(f"• Baseline (Climatologia Pura) : {rmse_baseline:.4f} mm/dia")
print(f"• Modelo LightGBM Multivariado : {rmse_modelo:.4f} mm/dia")
print(f"• Ganho de Acurácia            : {(rmse_baseline - rmse_modelo):.4f} mm/dia")
print("="*60)

# Salvar Artefatos
joblib.dump(model, "modelo_lightgbm.pkl")
joblib.dump(features_modelo, "features_list.pkl")
joblib.dump(clim_df, "climatologia_df.pkl")
print("\nModelo e lista de features salvos com sucesso!")
