# train_and_compare_models.py
# Requisitos: pandas, numpy, scikit-learn, xgboost, joblib, matplotlib, seaborn
# pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn

import os
import time
import joblib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import parallel_backend


# ===============================================================
# ----------------- 0. Configuración Global ---------------------
# ===============================================================
print("Iniciando script de entrenamiento y comparación de modelos...")

FEAT_CSV = "movies_dataset_fe_mlb_5actors.csv"        # CSV generado por el FE
PREPROCESSOR_JBL = "preprocessor_mlb_5actors.joblib" # joblib con 'preprocessor' y 'feature_cols' y 'target_col'
OUTPUT_MODELS_DIR = "models_saved"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_RATIO_WITHIN_TRAIN = 0.10  # para early stopping de XGBoost

# Configuración de clustering
K_RANGE = range(2, 10)           # k a evaluar (2..9)
SAMPLE_FOR_SIL = 10000           # tamaño de muestra para silhouette
PCA_VARIANCE = 0.95              # retener 95% de varianza con PCA

os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")
print(f"Directorio de salida de modelos y gráficos: {OUTPUT_MODELS_DIR}")


# ===============================================================
# ----------------- 1. Carga de Datos y Preprocessor ------------
# ===============================================================
print("\n===== 1. Carga de Datos y Preprocessor =====")

if not os.path.exists(FEAT_CSV):
    raise FileNotFoundError(f"No encontré el CSV de features: {FEAT_CSV}. Ejecutá antes el script de FE.")

if not os.path.exists(PREPROCESSOR_JBL):
    raise FileNotFoundError(f"No encontré el preprocessor: {PREPROCESSOR_JBL}. Ejecutá antes el script de FE.")

print(f"Cargando dataframe desde {FEAT_CSV}...")
df = pd.read_csv(FEAT_CSV)
print(f"Cargando preprocessor desde {PREPROCESSOR_JBL}...")
meta = joblib.load(PREPROCESSOR_JBL)

# Extraer componentes del preprocessor
preprocessor = meta.get('preprocessor')
feature_cols = meta.get('feature_cols')
target_col = meta.get('target_col', 'vote_average')
num_features = meta.get('num_features', []) # Para profiling de clusters
genre_cols = meta.get('genre_features', []) # Para profiling de clusters

if preprocessor is None or feature_cols is None:
    raise ValueError("El joblib de preprocessor no contiene 'preprocessor' o 'feature_cols'. Revisa el archivo.")

print(f"Dimensiones dataframe: {df.shape}")
print(f"Usando {len(feature_cols)} features (ej: {feature_cols[:3]}...)")
print(f"Target (Regresión): {target_col}")

# Asegurarnos de que las columnas existan en el dataframe
missing = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas en el CSV: {missing}")

# ===============================================================
# ----------------- 2. Preparación de Datos (Split) -------------
# ===============================================================
print("\n===== 2. Preparación de Datos (Split) =====")

# --- 2.1. Split para Regresión ---
X = df[feature_cols]
y = df[target_col]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Split Regresión -> Train full: {X_train_full.shape}, Test: {X_test.shape}")

# Sub-split para early stopping de XGBoost (Regresión)
X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
    X_train_full, y_train_full, test_size=VAL_RATIO_WITHIN_TRAIN, random_state=RANDOM_STATE
)
print(f"Split Regresión (XGB) -> Train: {X_train_reg.shape}, Val: {X_val_reg.shape}")


# ===============================================================
# ------------ 3. REGRESIÓN (Predicción de 'vote_average') ------
# ===============================================================
print("\n\n===== 3. REGRESIÓN (Predicción de 'vote_average') =====")

# --- 3.1. Definición de Modelos de Regresión ---
print("--- 3.1. Definición de Modelos de Regresión ---")
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
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE
    ))
])

# XGBoost (se entrena fuera del Pipeline para usar early stopping)
xgb_model = XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=50, n_jobs=-1,
    random_state=RANDOM_STATE, verbosity=1,
    objective='reg:squarederror'
)

# --- 3.2. Entrenamiento y Evaluación (Regresión) ---
print("\n--- 3.2. Entrenamiento y Evaluación (Regresión) ---")
results = []
predictions = {} # Guardar predicciones para gráficos

# 1) Modelos basados en Pipeline (Dummy, Ridge, RandomForest)
for name, pipe in models.items():
    model_path = os.path.join(OUTPUT_MODELS_DIR, f"{name}.joblib")
    train_time = 0

    if os.path.exists(model_path):
        print(f"\nModelo {name} ya existe. Cargando desde: {model_path}")
        pipe = joblib.load(model_path)
        train_time = np.nan # No se entrenó en esta ejecución
    else:
        print(f"\nEntrenando modelo: {name}")
        t0 = time.time()
        pipe.fit(X_train_reg, y_train_reg) # Usar X_train_reg/y_train_reg
        train_time = time.time() - t0
        print(f"Guardando modelo en: {model_path}")
        joblib.dump(pipe, model_path)

    # Evaluación (siempre se hace)
    y_pred = pipe.predict(X_test)
    predictions[name] = y_pred
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    print(f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}, Tiempo (s): {train_time:.1f}")
    results.append({
        'model': name, 'rmse': rmse, 'r2': r2, 'train_time_s': train_time
    })

# 2) XGBoost: transformar X y usar early stopping
print("\nPreparando datos transformados para XGBoost (Regresión)...")
# Usar el preprocessor fiteado (que está dentro del pipe 'ridge' o 'baseline', ya fiteado)
preprocessor_fitted = models['ridge'].named_steps['pre']
X_train_trans_reg = preprocessor_fitted.transform(X_train_reg)
X_val_trans_reg = preprocessor_fitted.transform(X_val_reg)
X_test_trans_reg = preprocessor_fitted.transform(X_test)


# ===============================================
# 3.2.1 Búsqueda de Hiperparámetros XGBoost
# ===============================================

# Conjuntos de parámetros a evaluar
xgb_params_list = [
    {"n_estimators": 500, "learning_rate": 0.1, "max_depth": 5},
    {"n_estimators": 800, "learning_rate": 0.05, "max_depth": 7},
    {"n_estimators": 1200, "learning_rate": 0.03, "max_depth": 8}
]

xgb_results = []
best_rmse = float("inf")
best_model = None
best_name = None

print("\n===== INICIO DE BÚSQUEDA XGBOOST (Regresión) =====")

for i, params in enumerate(xgb_params_list, start=1):
    print(f"\nEntrenando XGBoost versión {i} con parámetros: {params}")

    model = XGBRegressor(
        objective="reg:squarederror",
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        early_stopping_rounds=50,
        verbosity=0,
        **params
    )

    model.fit(
        X_train_trans_reg, y_train_reg,
        eval_set=[(X_val_trans_reg, y_val_reg)],
        verbose=False
    )

    y_pred = model.predict(X_test_trans_reg)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Versión {i} -> RMSE={rmse:.4f}, R²={r2:.4f}")

    xgb_results.append({
        "version": i,
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "rmse": rmse,
        "r2": r2
    })

    # Guardar cada modelo (opcional)
    joblib.dump(model, os.path.join(OUTPUT_MODELS_DIR, f"xgboost_v{i}.joblib"))

    # Actualizar mejor modelo
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_name = f"xgboost_v{i}"

# Guardar el mejor modelo
if best_model:
    joblib.dump(best_model, os.path.join(OUTPUT_MODELS_DIR, f"{best_name}_best.joblib"))
    print(f"\n✅ Mejor modelo: {best_name}_best.joblib (RMSE={best_rmse:.4f})")

# --------------------------------------------
# Guardar resultados y graficar comparaciones
# --------------------------------------------
xgb_results_df = pd.DataFrame(xgb_results)
xgb_results_df.to_csv(os.path.join(OUTPUT_MODELS_DIR, "xgboost_param_results.csv"), index=False)
print(f"\nResultados guardados en: models_saved/xgboost_param_results.csv")

# --- Gráfico RMSE ---
plt.figure(figsize=(8, 5))
sns.barplot(data=xgb_results_df, x="version", y="rmse", palette="Reds_r")
plt.title("Comparación de RMSE entre configuraciones XGBoost", fontsize=14)
plt.xlabel("Versión del modelo")
plt.ylabel("RMSE (menor es mejor)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_MODELS_DIR, "xgboost_rmse_comparison.png"))
plt.show()

# --- Gráfico R² ---
plt.figure(figsize=(8, 5))
sns.barplot(data=xgb_results_df, x="version", y="r2", palette="Greens_r")
plt.title("Comparación de R² entre configuraciones XGBoost", fontsize=14)
plt.xlabel("Versión del modelo")
plt.ylabel("R² (mayor es mejor)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_MODELS_DIR, "xgboost_r2_comparison.png"))
plt.show()

print("\n===== FIN DE BÚSQUEDA XGBOOST =====")
print(xgb_results_df)



model_path_xgb = os.path.join(OUTPUT_MODELS_DIR, "xgboost.joblib")
train_time_xgb = 0

if os.path.exists(model_path_xgb):
    print(f"Modelo xgboost ya existe. Cargando desde: {model_path_xgb}")
    xgb_model = joblib.load(model_path_xgb)
    train_time_xgb = np.nan
else:
    print("Entrenando XGBoost con early stopping...")
    t0 = time.time()
    xgb_model.fit(
        X_train_trans_reg, y_train_reg,
        eval_set=[(X_val_trans_reg, y_val_reg)],
        verbose=50
    )
    train_time_xgb = time.time() - t0
    print(f"Guardando modelo en: {model_path_xgb}")
    joblib.dump(xgb_model, model_path_xgb)

# Evaluación XGBoost
y_pred_xgb = xgb_model.predict(X_test_trans_reg)
predictions['xgboost'] = y_pred_xgb
rmse_xgb = float(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
r2_xgb = float(r2_score(y_test, y_pred_xgb))

print(f"XGBoost -> RMSE: {rmse_xgb:.4f}, R2: {r2_xgb:.4f}, Tiempo (s): {train_time_xgb:.1f}")
results.append({
    'model': 'xgboost', 'rmse': rmse_xgb, 'r2': r2_xgb, 'train_time_s': train_time_xgb
})

# --- 3.3. Resumen y Visualización (Regresión) ---
print("\n--- 3.3. Resumen y Visualización (Regresión) ---")
results_df = pd.DataFrame(results).sort_values('rmse').reset_index(drop=True)
print("\n===== RESULTADOS REGRESIÓN (ordenados por RMSE) =====")
print(results_df)

# Guardar resumen
results_df.to_csv(os.path.join(OUTPUT_MODELS_DIR, "regression_results.csv"), index=False)
print(f"Resumen de regresión guardado en: {os.path.join(OUTPUT_MODELS_DIR, 'regression_results.csv')}")

# Gráfico 1: Comparación RMSE
try:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='rmse', y='model', data=results_df.sort_values('rmse', ascending=False), orient='h', palette='Reds_r')
    plt.title('Comparación de Modelos (Regresión) - RMSE (Menor es mejor)', fontsize=16)
    plt.xlabel('Root Mean Squared Error (RMSE)', fontsize=12)
    plt.ylabel('Modelo', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_MODELS_DIR, 'regression_rmse_comparison.png')
    plt.savefig(plot_path)
    print(f"Gráfico de RMSE guardado en: {plot_path}")
    plt.close()

    # Gráfico 2: Comparación R2
    plt.figure(figsize=(10, 6))
    sns.barplot(x='r2', y='model', data=results_df.sort_values('r2', ascending=True), orient='h', palette='Greens_r')
    plt.title('Comparación de Modelos (Regresión) - R² (Mayor es mejor)', fontsize=16)
    plt.xlabel('R² Score', fontsize=12)
    plt.ylabel('Modelo', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_MODELS_DIR, 'regression_r2_comparison.png')
    plt.savefig(plot_path)
    print(f"Gráfico de R2 guardado en: {plot_path}")
    plt.close()

    # Gráfico 3: Actual vs Predicho (Mejor modelo)
    best_model_name = results_df.iloc[0]['model']
    y_pred_best = predictions[best_model_name]

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.5, s=30)
    # Línea de predicción perfecta
    min_val = min(y_test.min(), y_pred_best.min())
    max_val = max(y_test.max(), y_pred_best.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
    plt.title(f'Actual vs. Predicho (Mejor Modelo: {best_model_name})', fontsize=16)
    plt.xlabel('Valor Real (vote_average)', fontsize=12)
    plt.ylabel('Valor Predicho', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_MODELS_DIR, f'regression_actual_vs_pred_{best_model_name}.png')
    plt.savefig(plot_path)
    print(f"Gráfico Actual vs Predicho guardado en: {plot_path}")
    plt.close()

except Exception as e:
    print(f"Error al generar gráficos de regresión: {e}")


# ===============================================================
# ------------ 4. CLASIFICACIÓN (Predicción de 'rating_class') --
# ===============================================================
print("\n\n===== 4. CLASIFICACIÓN (Predicción de 'rating_class') =====")

# --- 4.1. Preparación de Datos (Clasificación) ---
print("--- 4.1. Preparación de Datos (Clasificación) ---")

def categorize_vote(v):
    if v < 5: return 0  # baja valoración
    elif v <= 7: return 1  # media valoración
    else: return 2  # alta valoración

df['rating_class'] = df[target_col].apply(categorize_vote)

# Redefinir X (mismas features) e y (nueva target)
X_clf = df[feature_cols]
y_clf = df['rating_class']

# Nuevo split estratificado para clasificación
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clf
)

print("\nDistribución de clases en train (Clasificación):")
print(y_train_clf.value_counts(normalize=True))
print("\nDistribución de clases en test (Clasificación):")
print(y_test_clf.value_counts(normalize=True))

# Sub-split para early stopping (Clasificación)
X_train_cl, X_val_cl, y_train_cl, y_val_cl = train_test_split(
    X_train_clf, y_train_clf, test_size=VAL_RATIO_WITHIN_TRAIN, random_state=RANDOM_STATE, stratify=y_train_clf
)
y_train_cl = y_train_cl.astype(int)
y_val_cl = y_val_cl.astype(int)

# --- 4.2. Definición de Modelos (Clasificación) ---
print("\n--- 4.2. Definición de Modelos (Clasificación) ---")
clf_models = {
    "baseline_dummy_clf": Pipeline([
        ('pre', preprocessor),
        ('clf', DummyClassifier(strategy='most_frequent'))
    ]),
    "logistic_regression": Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=1, random_state=RANDOM_STATE))
    ]),
    "random_forest_clf": Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=2,
            class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE
        ))
    ])
}

# XGBClassifier (fuera del pipeline)
xgb_clf = XGBClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, n_jobs=4,
    random_state=RANDOM_STATE, objective='multi:softmax',
    num_class=3, verbosity=0, early_stopping_rounds=30
)

# --- 4.3. Entrenamiento y Evaluación (Clasificación) ---
print("\n--- 4.3. Entrenamiento y Evaluación (Clasificación) ---")
clf_results = []
clf_predictions = {} # Guardar predicciones para gráficos

for name, model in clf_models.items():
    model_path = os.path.join(OUTPUT_MODELS_DIR, f"{name}.joblib")
    train_time = 0

    if os.path.exists(model_path):
        print(f"\nClasificador {name} ya existe. Cargando desde: {model_path}")
        model = joblib.load(model_path)
        train_time = np.nan
    else:
        print(f"\nEntrenando clasificador: {name}")
        t0 = time.time()
        with parallel_backend('threading'):
            model.fit(X_train_cl, y_train_cl) # Usar split de clasificación
        train_time = time.time() - t0
        print(f"Guardando modelo en: {model_path}")
        joblib.dump(model, model_path)

    # Evaluación (siempre se hace)
    y_pred = model.predict(X_test_clf)
    clf_predictions[name] = y_pred
    acc = accuracy_score(y_test_clf, y_pred)
    f1 = f1_score(y_test_clf, y_pred, average='macro')

    print(f"{name} -> Accuracy: {acc:.4f}, F1-macro: {f1:.4f}, Tiempo: {train_time:.1f}s")
    # print(classification_report(y_test_clf, y_pred, digits=3))
    # print("Matriz de confusión:\n", confusion_matrix(y_test_clf, y_pred))

    clf_results.append({
        'model': name, 'accuracy': acc, 'f1_macro': f1, 'train_time_s': train_time
    })

# ---------- XGBoost Clasificación ----------
print("\nPreparando datos transformados para XGBoost (Clasificación)...")
# Usar el preprocessor fiteado (que está dentro del pipe 'logistic_regression', ya fiteado)
preprocessor_clf_fitted = clf_models['logistic_regression'].named_steps['pre']
X_train_trans_clf = preprocessor_clf_fitted.transform(X_train_cl)
X_val_trans_clf = preprocessor_clf_fitted.transform(X_val_cl)
X_test_trans_clf = preprocessor_clf_fitted.transform(X_test_clf)

model_path_xgb_clf = os.path.join(OUTPUT_MODELS_DIR, "xgboost_classifier.joblib")
train_time_xgb_clf = 0

if os.path.exists(model_path_xgb_clf):
    print(f"Modelo xgboost_classifier ya existe. Cargando desde: {model_path_xgb_clf}")
    xgb_clf = joblib.load(model_path_xgb_clf)
    train_time_xgb_clf = np.nan
else:
    print("Entrenando XGBoostClassifier con early stopping...")
    t0 = time.time()
    xgb_clf.fit(
        X_train_trans_clf, y_train_cl,
        eval_set=[(X_val_trans_clf, y_val_cl)],
        verbose=False
    )
    train_time_xgb_clf = time.time() - t0
    print(f"Guardando modelo en: {model_path_xgb_clf}")
    joblib.dump(xgb_clf, model_path_xgb_clf)

# Evaluación XGBoost
y_pred_xgb_clf = xgb_clf.predict(X_test_trans_clf)
clf_predictions['xgboost_classifier'] = y_pred_xgb_clf
acc_xgb = accuracy_score(y_test_clf, y_pred_xgb_clf)
f1_xgb = f1_score(y_test_clf, y_pred_xgb_clf, average='macro')

print(f"XGBoostClassifier -> Accuracy: {acc_xgb:.4f}, F1-macro: {f1_xgb:.4f}, Tiempo: {train_time_xgb_clf:.1f}s")
# print(classification_report(y_test_clf, y_pred_xgb_clf, digits=3))
# print("Matriz de confusión:\n", confusion_matrix(y_test_clf, y_pred_xgb_clf))

clf_results.append({
    'model': 'xgboost_classifier', 'accuracy': acc_xgb, 'f1_macro': f1_xgb, 'train_time_s': train_time_xgb_clf
})


# --- 4.4. Resumen y Visualización (Clasificación) ---
print("\n--- 4.4. Resumen y Visualización (Clasificación) ---")
clf_results_df = pd.DataFrame(clf_results).sort_values('f1_macro', ascending=False).reset_index(drop=True)
print("\n===== RESULTADOS CLASIFICACIÓN (ordenados por F1-macro) =====")
print(clf_results_df)

# Guardar resumen
clf_results_df.to_csv(os.path.join(OUTPUT_MODELS_DIR, "classification_results.csv"), index=False)
print(f"Resumen de clasificación guardado en: {os.path.join(OUTPUT_MODELS_DIR, 'classification_results.csv')}")

# Gráfico 1: Comparación F1-macro
try:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='f1_macro', y='model', data=clf_results_df.sort_values('f1_macro', ascending=True), orient='h', palette='Blues_r')
    plt.title('Comparación de Modelos (Clasificación) - F1-macro (Mayor es mejor)', fontsize=16)
    plt.xlabel('F1-score (macro)', fontsize=12)
    plt.ylabel('Modelo', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_MODELS_DIR, 'classification_f1_comparison.png')
    plt.savefig(plot_path)
    print(f"Gráfico de F1 guardado en: {plot_path}")
    plt.close()

    # Gráfico 2: Comparación Accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(x='accuracy', y='model', data=clf_results_df.sort_values('accuracy', ascending=True), orient='h', palette='Oranges_r')
    plt.title('Comparación de Modelos (Clasificación) - Accuracy (Mayor es mejor)', fontsize=16)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Modelo', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_MODELS_DIR, 'classification_accuracy_comparison.png')
    plt.savefig(plot_path)
    print(f"Gráfico de Accuracy guardado en: {plot_path}")
    plt.close()

    # Gráfico 3: Matriz de Confusión (Mejor modelo)
    best_clf_name = clf_results_df.iloc[0]['model']
    y_pred_best_clf = clf_predictions[best_clf_name]
    cm = confusion_matrix(y_test_clf, y_pred_best_clf)
    class_names = ['Baja', 'Media', 'Alta'] # Basado en 0, 1, 2

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    plt.title(f'Matriz de Confusión (Mejor Modelo: {best_clf_name})', fontsize=16)
    plt.xlabel('Predicho', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_MODELS_DIR, f'classification_confusion_matrix_{best_clf_name}.png')
    plt.savefig(plot_path)
    print(f"Gráfico de Matriz de Confusión guardado en: {plot_path}")
    plt.close()

except Exception as e:
    print(f"Error al generar gráficos de clasificación: {e}")


# ===============================================================
# ----------------- 5. CLUSTERING (KMeans) ----------------------
# ===============================================================
print("\n\n===== 5. CLUSTERING (KMeans) =====")

# --- 5.1. Preparación de Datos (PCA) ---
print("--- 5.1. Preparación de Datos (PCA) ---")

# Usar el preprocessor ya fiteado (p.ej. de regresión) para transformar TODO el dataset
print("Transformando features con el preprocessor (esto puede tardar)...")
X_full = df[feature_cols]
X_trans = preprocessor_fitted.transform(X_full)  # numpy array
print("Matriz transformada:", X_trans.shape)

# Reducir dimensionalidad con PCA
print(f"Aplicando PCA para retener {int(PCA_VARIANCE*100)}% de varianza...")
pca = PCA(n_components=PCA_VARIANCE, svd_solver='full', random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_trans)
print("Dimensiones después de PCA:", X_pca.shape)
print(f"Varianza explicada acumulada: {pca.explained_variance_ratio_.sum():.4f}")

# --- 5.2. Búsqueda de K Óptimo ---
print("\n--- 5.2. Búsqueda de K Óptimo ---")
curves_path = os.path.join(OUTPUT_MODELS_DIR, "kmeans_curves.csv")
best_k = 4 # Default por si falla

if os.path.exists(curves_path):
    print(f"Curvas de KMeans ya existen. Cargando desde: {curves_path}")
    curve_df = pd.read_csv(curves_path)
    inertias = curve_df['inertia'].tolist()
    sil_scores = curve_df['silhouette'].tolist()
else:
    print("Calculando curvas de Inercia y Silhouette...")
    inertias = []
    sil_scores = []

    # Preparar muestra para silhouette
    n_samples = X_pca.shape[0]
    use_sil_sample = min(SAMPLE_FOR_SIL, n_samples)
    sample_idx = None
    if n_samples > use_sil_sample:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(n_samples, use_sil_sample, replace=False)
        X_sil_sample = X_pca[sample_idx]
    else:
        X_sil_sample = X_pca

    print(f"Evaluando k en rango: {list(K_RANGE)} (usando muestra de {use_sil_sample} para silhouette)")
    for k in K_RANGE:
        print(f"  - k = {k} ...", end="")
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_pca)
        inertias.append(kmeans.inertia_)
        # silhouette sobre muestra
        labels_sample = kmeans.labels_[sample_idx] if sample_idx is not None else kmeans.labels_
        try: sil = silhouette_score(X_sil_sample, labels_sample)
        except Exception: sil = float('nan')
        sil_scores.append(sil)
        print(f" inertia={kmeans.inertia_:.1f}, silhouette={sil:.4f}")

    # Guardar curvas
    curve_df = pd.DataFrame({'k': list(K_RANGE), 'inertia': inertias, 'silhouette': sil_scores})
    curve_df.to_csv(curves_path, index=False)
    print(f"Curvas guardadas en {curves_path}")

# Elegir k sugerido automáticamente (máx silhouette)
valid_indices = [i for i, s in enumerate(sil_scores) if not math.isnan(s)]
if len(valid_indices) > 0:
    best_idx = valid_indices[np.argmax([sil_scores[i] for i in valid_indices])]
    best_k = list(K_RANGE)[best_idx]
    print(f"\nK sugerido por silhouette: {best_k} (silhouette={sil_scores[best_idx]:.4f})")
else:
    print("\nNo se pudo calcular silhouette, usando k=4 por defecto.")
    best_k = 4

# --- 5.3. Entrenamiento Final y Guardado ---
print("\n--- 5.3. Entrenamiento Final y Guardado ---")
final_model_path = os.path.join(OUTPUT_MODELS_DIR, f"kmeans_k{best_k}.joblib")
labels = None

if os.path.exists(final_model_path):
    print(f"Modelo KMeans (k={best_k}) ya existe. Cargando...")
    kmeans_data = joblib.load(final_model_path)
    kmeans_final = kmeans_data['kmeans']
    # 'pca' también está en kmeans_data, pero ya lo tenemos fiteado
    labels = kmeans_final.predict(X_pca) # Asignar labels a los datos PCA
else:
    print(f"Entrenando KMeans final con k={best_k} ...")
    kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    kmeans_final.fit(X_pca)
    labels = kmeans_final.labels_
    # Guardar modelo KMeans y PCA
    joblib.dump({'kmeans': kmeans_final, 'pca': pca}, final_model_path)
    print(f"KMeans y PCA guardados en {final_model_path}")

# Asignar clusters al DataFrame original y guardar
df_clusters = df.copy()
df_clusters['kmeans_cluster'] = labels
out_clusters_csv = os.path.join(OUTPUT_MODELS_DIR, f"movies_with_kmeans_k{best_k}.csv")
df_clusters.to_csv(out_clusters_csv, index=False)
print(f"Dataframe con clusters guardado en: {out_clusters_csv}")

# --- 5.4. Perfilado de Clusters (Opcional, intensivo en I/O) ---
# (Se mantiene la lógica original de guardado de TXT y CSV)
print("\n--- 5.4. Perfilado de Clusters ---")
try:
    # 7a) tamaño y medias numéricas
    cluster_stats = df_clusters.groupby('kmeans_cluster')[num_features].agg(['count','mean','std']).transpose()
    cluster_stats.to_csv(os.path.join(OUTPUT_MODELS_DIR, f"kmeans_cluster_numeric_stats_k{best_k}.csv"))
    print("Stats numéricas por cluster guardadas.")

    # 7b) Top géneros
    genre_summary = {}
    with open(os.path.join(OUTPUT_MODELS_DIR, f"kmeans_top_genres_k{best_k}.txt"), 'w', encoding='utf-8') as f:
        for c in range(best_k):
            sub = df_clusters[df_clusters['kmeans_cluster'] == c]
            sums = sub[genre_cols].sum().sort_values(ascending=False)
            genre_summary[c] = sums.head(10)
            f.write(f"Cluster {c} (N={len(sub)}) - Top géneros:\n{sums.head(10).to_string()}\n\n")
    print("Resumen de géneros por cluster guardado.")

    # 7c) Top actores/directores
    actor_cols = [c for c in feature_cols if c.startswith('actor_')]
    director_col = 'director' if 'director' in feature_cols else None
    with open(os.path.join(OUTPUT_MODELS_DIR, f"kmeans_top_actors_directors_k{best_k}.txt"), 'w', encoding='utf-8') as f:
        for c in range(best_k):
            sub = df_clusters[df_clusters['kmeans_cluster'] == c]
            f.write(f"Cluster {c} (N={len(sub)})\n")
            actors_series = pd.Series(sub[actor_cols].values.ravel()).dropna()
            f.write(f"Top actores:\n{actors_series.value_counts().head(10).to_string()}\n")
            if director_col:
                f.write(f"Top directores:\n{sub[director_col].value_counts().head(10).to_string()}\n")
            f.write("\n-----------------\n\n")
    print("Resumen de actores/directores por cluster guardado.")

    # 8) Resumen final
    sil_full = silhouette_score(X_pca, labels)
    summary = [{'cluster': c, 'n_items': int((df_clusters['kmeans_cluster'] == c).sum())} for c in range(best_k)]
    summary_df = pd.DataFrame(summary)
    summary_df['total'] = len(df_clusters)
    summary_df['pct'] = summary_df['n_items'] / summary_df['total']
    summary_df['global_silhouette'] = sil_full
    summary_df.to_csv(os.path.join(OUTPUT_MODELS_DIR, f"kmeans_summary_k{best_k}.csv"), index=False)
    print(f"Resumen final de clustering guardado (Silhouette global: {sil_full:.4f}).")

except Exception as e:
    print(f"Error durante el perfilado de clusters: {e}")


# --- 5.5. Visualización de Resultados (Clustering) ---
print("\n--- 5.5. Visualización de Resultados (Clustering) ---")

try:
    # Gráfico 1: Curva del Codo y Silhouette
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx() # Eje Y secundario
    sns.lineplot(data=curve_df, x='k', y='inertia', marker='o', color='b', ax=ax1, label='Inercia (Codo)')
    sns.lineplot(data=curve_df, x='k', y='silhouette', marker='s', color='r', ax=ax2, label='Silhouette')
    ax1.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inercia', color='b', fontsize=12)
    ax2.set_ylabel('Score de Silhouette', color='r', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.axvline(best_k, color='gray', linestyle='--', label=f'Mejor k = {best_k} (por Silhouette)')
    plt.title('Curva del Codo y Score de Silhouette vs. k', fontsize=16)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(OUTPUT_MODELS_DIR, 'clustering_elbow_silhouette_curves.png')
    plt.savefig(plot_path)
    print(f"Gráfico de curvas K-Means guardado en: {plot_path}")
    plt.close()

    # Gráfico 2: Dispersión de Clusters (PCA)
    # Usar una muestra si el dataset es muy grande para graficar
    if X_pca.shape[0] > 20000:
        print("Dataset grande, tomando muestra de 20000 para gráfico de dispersión PCA...")
        rng_plot = np.random.RandomState(RANDOM_STATE)
        plot_idx = rng_plot.choice(X_pca.shape[0], 20000, replace=False)
        X_pca_sample = X_pca[plot_idx]
        labels_sample = labels[plot_idx]
    else:
        X_pca_sample = X_pca
        labels_sample = labels

    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        x=X_pca_sample[:, 0], y=X_pca_sample[:, 1],
        hue=labels_sample,
        palette=sns.color_palette('viridis', n_colors=best_k),
        alpha=0.6, s=30, legend='full'
    )
    plt.title(f'Visualización de Clusters (k={best_k}) - Primeros 2 Componentes PCA', fontsize=16)
    plt.xlabel('Componente Principal 1', fontsize=12)
    plt.ylabel('Componente Principal 2', fontsize=12)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para la leyenda
    plot_path = os.path.join(OUTPUT_MODELS_DIR, f'clustering_pca_scatter_k{best_k}.png')
    plt.savefig(plot_path)
    print(f"Gráfico de dispersión PCA guardado en: {plot_path}")
    plt.close()

except Exception as e:
    print(f"Error al generar gráficos de clustering: {e}")


print("\n\n===================== SCRIPT COMPLETADO =====================")
print(f"Todos los modelos, resultados y gráficos se han guardado en: {OUTPUT_MODELS_DIR}")