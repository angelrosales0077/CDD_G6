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
from joblib import parallel_backend


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



# ===============================================================
# 🔹 SECCIÓN DE CLASIFICACIÓN (cumple con la consigna)
# ===============================================================
print("\n\n===================== CLASIFICACIÓN =====================")

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ---------- Crear variable categórica a partir de 'vote_average' ----------
def categorize_vote(v):
    if v < 5:
        return 0  # baja valoración
    elif v <= 7:
        return 1  # media valoración
    else:
        return 2  # alta valoración

df['rating_class'] = df[target_col].apply(categorize_vote)

# ---------- Redefinir X e y ----------
X = df[feature_cols]
y = df['rating_class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print("\nDistribución de clases en train:")
print(y_train.value_counts(normalize=True))

# ---------- Modelos de clasificación ----------
clf_models = {
    "baseline_dummy": Pipeline([
        ('pre', preprocessor),
        ('clf', DummyClassifier(strategy='most_frequent'))
    ]),
    "logistic_regression": Pipeline([
        ('pre', preprocessor),
        # Evitar uso de procesos en Windows: usar n_jobs=1
        ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', n_jobs=1))
    ]),
    "random_forest_clf": Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])
}

# XGBClassifier (fuera del pipeline)
xgb_clf = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    random_state=RANDOM_STATE,
    objective='multi:softmax',
    num_class=3,
    verbosity=0
)

# ---------- Entrenamiento y evaluación ----------
clf_results = []

for name, model in clf_models.items():
    print(f"\nEntrenando clasificador: {name}")
    t0 = time.time()
    # Forzar backend de joblib a 'threading' para evitar spawn de procesos problemáticos
    with parallel_backend('threading'):
        model.fit(X_train, y_train)

    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"{name} -> Accuracy: {acc:.4f}, F1-macro: {f1:.4f}, Tiempo: {train_time:.1f}s")
    print(classification_report(y_test, y_pred, digits=3))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    clf_results.append({
        'model': name,
        'accuracy': acc,
        'f1_macro': f1,
        'train_time_s': train_time
    })
    joblib.dump(model, os.path.join(OUTPUT_MODELS_DIR, f"{name}.joblib"))

# ---------- XGBoost ----------
print("\nEntrenando XGBoostClassifier...")

# Crear validation set específico para clasificación (evita usar y_val de la sección de regresión)
X_train_cl, X_val_cl, y_train_cl, y_val_cl = train_test_split(
    X_train, y_train, test_size=VAL_RATIO_WITHIN_TRAIN, random_state=RANDOM_STATE, stratify=y_train
)

# Asegurar que las etiquetas sean enteros en el rango [0, num_class)
y_train_cl = y_train_cl.astype(int)
y_val_cl = y_val_cl.astype(int)

# Transformar datos con el preprocessor fiteado
X_train_trans = preprocessor.transform(X_train_cl)
X_val_trans = preprocessor.transform(X_val_cl)
X_test_trans = preprocessor.transform(X_test)

t0 = time.time()
xgb_clf.fit(
    X_train_trans, y_train_cl,
    eval_set=[(X_val_trans, y_val_cl)],
    early_stopping_rounds=30,
    verbose=False
)
train_time = time.time() - t0

y_pred_xgb = xgb_clf.predict(X_test_trans)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')

print(f"XGBoostClassifier -> Accuracy: {acc_xgb:.4f}, F1-macro: {f1_xgb:.4f}, Tiempo: {train_time:.1f}s")
print(classification_report(y_test, y_pred_xgb, digits=3))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_xgb))

clf_results.append({
    'model': 'xgboost_classifier',
    'accuracy': acc_xgb,
    'f1_macro': f1_xgb,
    'train_time_s': train_time
})
joblib.dump(xgb_clf, os.path.join(OUTPUT_MODELS_DIR, "xgboost_classifier.joblib"))

# ---------- Resumen ----------
clf_results_df = pd.DataFrame(clf_results).sort_values('f1_macro', ascending=False)
print("\n===== RESULTADOS DE CLASIFICACIÓN =====")
print(clf_results_df)

clf_results_df.to_csv(os.path.join(OUTPUT_MODELS_DIR, "classification_results.csv"), index=False)
print(f"\nGuardado resumen en: {os.path.join(OUTPUT_MODELS_DIR, 'classification_results.csv')}")


# ============================
# 🔹 SECCIÓN DE CLUSTERING (KMEANS)
# ============================
print("\n\n===================== CLUSTERING (KMEANS) =====================")

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math

# Parámetros
K_RANGE = range(2, 10)           # k a evaluar (2..9)
SAMPLE_FOR_SIL = 10000           # tamaño de muestra para silhouette (ajustable)
PCA_VARIANCE = 0.95              # retener 95% de varianza con PCA
RANDOM_STATE = RANDOM_STATE

# 1) Preparamos la matriz transformada (numérica) usando el preprocessor fiteado
print("Transformando features con el preprocessor (esto puede tardar)...")
X_full = df[feature_cols]
X_trans = preprocessor.transform(X_full)  # numpy array

print("Matrice transformada:", X_trans.shape)

# 2) Reducimos dimensionalidad con PCA para acelerar clustering
print(f"Aplicando PCA para retener {int(PCA_VARIANCE*100)}% de varianza...")
pca = PCA(n_components=PCA_VARIANCE, svd_solver='full', random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_trans)
print("Dimensiones después de PCA:", X_pca.shape)
print("Varianza explicada acumulada (último componente):", pca.explained_variance_ratio_.sum())

# 3) Curva del codo (inertia) y silhouette (en muestra)
inertias = []
sil_scores = []

# Para silhouette calculamos sobre una muestra aleatoria si el dataset es grande
n_samples = X_pca.shape[0]
use_sil_sample = min(SAMPLE_FOR_SIL, n_samples)
sample_idx = None
if n_samples > use_sil_sample:
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(n_samples, use_sil_sample, replace=False)
    X_sil_sample = X_pca[sample_idx]
else:
    X_sil_sample = X_pca

print("\nEvaluando k en rango:", list(K_RANGE))
for k in K_RANGE:
    print(f"  - k = {k} ...", end="")
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_pca)  # ajustamos sobre todos para inertia fiable
    inertias.append(kmeans.inertia_)
    # silhouette sobre muestra
    labels_sample = kmeans.labels_[sample_idx] if sample_idx is not None else kmeans.labels_
    try:
        sil = silhouette_score(X_sil_sample, labels_sample)
    except Exception as e:
        sil = float('nan')
    sil_scores.append(sil)
    print(f" inertia={kmeans.inertia_:.1f}, silhouette={sil:.4f}")

# 4) Elegir k sugerido automáticamente (máx silhouette)
valid_indices = [i for i, s in enumerate(sil_scores) if not math.isnan(s)]
if len(valid_indices) > 0:
    best_idx = valid_indices[np.argmax([sil_scores[i] for i in valid_indices])]
    best_k = list(K_RANGE)[best_idx]
    print(f"\nK sugerido por silhouette: {best_k} (silhouette={sil_scores[best_idx]:.4f})")
else:
    # fallback: elegir k por el "codo" (mínimo decremento relativo) - sencillo heurístico
    decrements = np.diff(inertias)
    rel_dec = decrements / inertias[:-1]
    best_idx = np.argmin(rel_dec) + 1
    best_k = list(K_RANGE)[best_idx]
    print(f"\nK sugerido por heurística de codo: {best_k}")

# Guardar curvas para análisis (CSV)
curve_df = pd.DataFrame({
    'k': list(K_RANGE),
    'inertia': inertias,
    'silhouette': sil_scores
})
curve_df.to_csv(os.path.join(OUTPUT_MODELS_DIR, "kmeans_curves.csv"), index=False)
print(f"Curvas guardadas en {os.path.join(OUTPUT_MODELS_DIR, 'kmeans_curves.csv')}")

# 5) Entrenar KMeans final con best_k
print(f"\nEntrenando KMeans final con k={best_k} ...")
kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
kmeans_final.fit(X_pca)  # se ajusta sobre la matriz PCA
labels = kmeans_final.labels_

# Guardar modelo KMeans y PCA
joblib.dump({'kmeans': kmeans_final, 'pca': pca}, os.path.join(OUTPUT_MODELS_DIR, f"kmeans_k{best_k}.joblib"))
print(f"KMeans y PCA guardados en {OUTPUT_MODELS_DIR}")

# 6) Asignar clusters al DataFrame original y guardar
df_clusters = df.copy()
df_clusters['kmeans_cluster'] = labels
out_clusters_csv = os.path.join(OUTPUT_MODELS_DIR, f"movies_with_kmeans_k{best_k}.csv")
df_clusters.to_csv(out_clusters_csv, index=False)
print(f"Dataframe con clusters guardado en: {out_clusters_csv}")

# 7) Perfilado de clusters: tamaño, medias numéricas, top géneros, top actores/directores
print("\nPerfilando clusters...")

# 7a) tamaño y medias numéricas (usar columnas numéricas del meta)
num_features = meta.get('num_features', [])
cluster_stats = df_clusters.groupby('kmeans_cluster')[num_features].agg(['count','mean','std']).transpose()
# Guardar stats numéricas
cluster_stats.to_csv(os.path.join(OUTPUT_MODELS_DIR, f"kmeans_cluster_numeric_stats_k{best_k}.csv"))
print("Stats numéricas por cluster guardadas.")

# 7b) Top géneros por cluster (las columnas de genre están en genre_features en meta)
genre_cols = meta.get('genre_features', [])
genre_summary = {}
for c in range(best_k):
    sub = df_clusters[df_clusters['kmeans_cluster'] == c]
    # sumar las columnas binarias de género para saber qué géneros predominan
    sums = sub[genre_cols].sum().sort_values(ascending=False)
    genre_summary[c] = sums.head(10)

# Guardar resumen de géneros legible
with open(os.path.join(OUTPUT_MODELS_DIR, f"kmeans_top_genres_k{best_k}.txt"), 'w', encoding='utf-8') as f:
    for c, s in genre_summary.items():
        f.write(f"Cluster {c} (N={len(df_clusters[df_clusters['kmeans_cluster']==c])}) - Top géneros:\n")
        f.write(s.to_string())
        f.write("\n\n")
print("Resumen de géneros por cluster guardado.")

# 7c) Top actores/directores por cluster (usando columnas actor_1..actor_5 y director)
actor_cols = [c for c in feature_cols if c.startswith('actor_')]
director_col = 'director' if 'director' in feature_cols else None

actor_summary = {}
director_summary = {}

for c in range(best_k):
    sub = df_clusters[df_clusters['kmeans_cluster'] == c]
    # actores: contamos frecuencias concatenando las columnas actor_1..actor_5 y contando valores
    actors_series = pd.Series(sub[actor_cols].values.ravel()).dropna()
    actors_top = actors_series.value_counts().head(10)
    actor_summary[c] = actors_top

    # director
    if director_col:
        dirs_top = sub[director_col].value_counts().head(10)
        director_summary[c] = dirs_top

# Guardar resúmenes de actores/directores
with open(os.path.join(OUTPUT_MODELS_DIR, f"kmeans_top_actors_directors_k{best_k}.txt"), 'w', encoding='utf-8') as f:
    for c in range(best_k):
        f.write(f"Cluster {c} (N={len(df_clusters[df_clusters['kmeans_cluster']==c])})\n")
        f.write("Top actores:\n")
        f.write(actor_summary[c].to_string())
        f.write("\n")
        if director_col:
            f.write("Top directores:\n")
            f.write(director_summary[c].to_string())
            f.write("\n")
        f.write("\n-----------------\n\n")
print("Resumen de actores/directores por cluster guardado.")

# 8) Guardar resumen final con tamaño y silhouette local (silhouette promedio por cluster si se puede calcular)
try:
    sil_full = silhouette_score(X_pca, labels)
except:
    sil_full = float('nan')

summary = []
for c in range(best_k):
    n_c = int((df_clusters['kmeans_cluster'] == c).sum())
    summary.append({
        'cluster': c,
        'n_items': n_c
    })

summary_df = pd.DataFrame(summary)
summary_df['total'] = len(df_clusters)
summary_df['pct'] = summary_df['n_items'] / summary_df['total']
summary_df['global_silhouette'] = sil_full

summary_df.to_csv(os.path.join(OUTPUT_MODELS_DIR, f"kmeans_summary_k{best_k}.csv"), index=False)
print(f"Resumen final guardado en: {os.path.join(OUTPUT_MODELS_DIR, f'kmeans_summary_k{best_k}.csv')}")

print("\nClusterización completada. Revisá los archivos en 'models_saved/' para los resultados y resúmenes.")

# models_saved/kmeans_k{best_k}.joblib — KMeans + PCA (serializados).

# models_saved/movies_with_kmeans_k{best_k}.csv — dataset con columna kmeans_cluster.

# models_saved/kmeans_curves.csv — inertia y silhouette por k evaluado.

# models_saved/kmeans_summary_k{best_k}.csv — tamaños y % por cluster, silhouette global.

# models_saved/kmeans_cluster_numeric_stats_k{best_k}.csv — medias/desv por cluster (numéricas).

# models_saved/kmeans_top_genres_k{best_k}.txt — top géneros por cluster.

# models_saved/kmeans_top_actors_directors_k{best_k}.txt — top actores/directores por cluster.