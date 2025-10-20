# train_and_compare_models.py
# Requisitos: pandas, numpy, scikit-learn, xgboost, joblib
# pip install pandas numpy scikit-learn xgboost joblib

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ----------------- Config -----------------
FEAT_CSV = "movies_dataset_fe_mlb_5actors.csv"        # CSV generado por el FE
PREPROCESSOR_JBL = "preprocessor_mlb_5actors.joblib" # joblib con 'preprocessor' y 'feature_cols' y 'target_col'
OUTPUT_MODELS_DIR = "models_saved"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_RATIO_WITHIN_TRAIN = 0.10  # para early stopping de XGBoost
# ------------------------------------------

os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)

# ---------- Cargar recursos ----------
if not os.path.exists(FEAT_CSV):
    raise FileNotFoundError(f"No encontré el CSV de features: {FEAT_CSV}. Ejecutá antes el script de FE.")

if not os.path.exists(PREPROCESSOR_JBL):
    raise FileNotFoundError(f"No encontré el preprocessor: {PREPROCESSOR_JBL}. Ejecutá antes el script de FE.")

print("Cargando dataframe y preprocessor...")
df = pd.read_csv(FEAT_CSV)
meta = joblib.load(PREPROCESSOR_JBL)

# El joblib debe contener al menos estas claves (según el script anterior):
preprocessor = meta.get('preprocessor')
feature_cols = meta.get('feature_cols')
target_col = meta.get('target_col', 'vote_average')

if preprocessor is None or feature_cols is None:
    raise ValueError("El joblib de preprocessor no contiene 'preprocessor' o 'feature_cols'. Revisa el archivo.")

print(f"Dimensiones dataframe: {df.shape}")
print(f"Usando features: {feature_cols}")
print(f"Target: {target_col}")

# Asegurarnos de que las columnas existan en el dataframe
missing = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas en el CSV: {missing}")

# ---------- Train/test split ----------
X = df[feature_cols]
y = df[target_col]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train full: {X_train_full.shape}, Test: {X_test.shape}")

# Para XGBoost usaremos un validation set sacado del train_full
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=VAL_RATIO_WITHIN_TRAIN, random_state=RANDOM_STATE
)
print(f"Train (para fit): {X_train.shape}, Val (para early stop): {X_val.shape}")

# ---------- Definición de modelos (parámetros iniciales) ----------
# Nota: usamos Pipeline(preprocessor + estimator) para Ridge y RandomForest.
models = {}

models['baseline_dummy'] = Pipeline([
    ('pre', preprocessor),
    ('model', DummyRegressor(strategy='mean'))
])

models['ridge'] = Pipeline([
    ('pre', preprocessor),
    ('model', Ridge(alpha=1.0, random_state=RANDOM_STATE))
])

models['random_forest'] = Pipeline([
    ('pre', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])

# XGBoost lo entrenamos fuera del Pipeline (transformamos con preprocessor)
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=1,
    objective='reg:squarederror'
)

# ---------- Entrenamiento y evaluación ----------
results = []

# 1) Modelos basados en Pipeline (Dummy, Ridge, RandomForest)
for name, pipe in models.items():
    print(f"\nEntrenando modelo: {name}")
    t0 = time.time()
    pipe.fit(X_train, y_train)  # pipeline aplicará preprocessor internamente
    train_time = time.time() - t0

    # Predicción en test (pipeline hace transform automáticamente)
    y_pred = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    print(f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}, Tiempo (s): {train_time:.1f}")
    results.append({
        'model': name,
        'rmse': rmse,
        'r2': r2,
        'train_time_s': train_time
    })

    # Guardar modelo
    joblib.dump(pipe, os.path.join(OUTPUT_MODELS_DIR, f"{name}.joblib"))
    print(f"Guardado: {os.path.join(OUTPUT_MODELS_DIR, f'{name}.joblib')}")

# 2) XGBoost: transformar X con preprocessor y usar early stopping
print("\nPreparando datos transformados para XGBoost (usa preprocessor fiteado)...")
# IMPORTANTE: preprocessor espera columnas en el mismo orden que feature_cols
X_train_trans = preprocessor.transform(X_train)
X_val_trans = preprocessor.transform(X_val)
X_test_trans = preprocessor.transform(X_test)

print("Entrenando XGBoost con early stopping...")
t0 = time.time()
xgb_model.fit(
    X_train_trans, y_train,
    eval_set=[(X_val_trans, y_val)],
    verbose=50
)
train_time = time.time() - t0

y_pred_xgb = xgb_model.predict(X_test_trans)
rmse_xgb = float(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
r2_xgb = float(r2_score(y_test, y_pred_xgb))

print(f"XGBoost -> RMSE: {rmse_xgb:.4f}, R2: {r2_xgb:.4f}, Tiempo (s): {train_time:.1f}")

results.append({
    'model': 'xgboost',
    'rmse': rmse_xgb,
    'r2': r2_xgb,
    'train_time_s': train_time
})

# Guardar XGBoost
joblib.dump(xgb_model, os.path.join(OUTPUT_MODELS_DIR, "xgboost.joblib"))
print(f"Guardado: {os.path.join(OUTPUT_MODELS_DIR, 'xgboost.joblib')}")

# ---------- Resumen de resultados ----------
results_df = pd.DataFrame(results).sort_values('rmse').reset_index(drop=True)
print("\n===== RESULTADOS (ordenados por RMSE ascendente) =====")
print(results_df)

# Guardar resumen
results_df.to_csv(os.path.join(OUTPUT_MODELS_DIR, "results_summary.csv"), index=False)
print(f"\nResumen guardado en: {os.path.join(OUTPUT_MODELS_DIR, 'results_summary.csv')}")
