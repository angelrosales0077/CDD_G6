# feature_engineering_updated.py
# Requisitos: pandas numpy scikit-learn joblib
# pip install pandas numpy scikit-learn joblib

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# ---------------------------
# Config
# ---------------------------
INPUT_CSV = "movies_dataset_reducido.csv"   # ajustá ruta
OUTPUT_CSV = "movies_dataset_fe_mlb_5actors.csv"
PREPROCESSOR_PATH = "preprocessor_mlb_5actors.joblib"

MIN_FREQ = 50   # categorías con < MIN_FREQ apariciones -> 'Other' (ajustable)

# ---------------------------
# Utilidades
# ---------------------------
def safe_split_first(s):
    """
    Toma una celda tipo "Drama, Romance" o "Anja Schüte, Valérie Dumas"
    y devuelve una lista de items limpios: ['Drama', 'Romance'].
    Si s es NaN/None devuelve [].
    """
    if pd.isna(s):
        return []
    # Convertir a str por si viene como float/int accidentalmente
    s_str = str(s).strip()
    if s_str == "":
        return []
    # Split simple por coma
    parts = [p.strip() for p in s_str.split(',') if p.strip() != ""]
    return parts

def top_k_or_minfreq_replace(series, min_freq=MIN_FREQ):
    """
    Reemplaza en 'series' las categorías con frecuencia < min_freq por 'Other'.
    Devuelve la serie transformada.
    """
    freq = series.value_counts(dropna=False)
    # Si la categoría es NaN la dejamos como NaN (o podríamos forzar 'Other')
    rare = freq[freq < min_freq].index
    return series.apply(lambda x: 'Other' if (pd.isna(x) == False and x in rare) else x)

# ---------------------------
# Feature engineering principal
# ---------------------------
def engineer_features(df, min_freq=MIN_FREQ, extract_actors_n=5):
    # Evitar duplicados / fuga: si existe averageRating (igual a vote_average) la eliminamos
    if 'averageRating' in df.columns:
        df = df.copy()  # trabajar sobre copia
        df.drop(columns=['averageRating'], inplace=True)

    # 1) release_date -> release_year, release_month
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df.drop(columns=['release_date'], inplace=True)

    # 2) Genres -> usar MultiLabelBinarizer (soporta multi-etiquetas)
    # Primero obtenemos listas con safe_split_first
    df['genres_list'] = df['genres'].apply(safe_split_first)
    mlb = MultiLabelBinarizer(sparse_output=False)
    # Fit-transform (esto crea columnas para cada género encontrado)
    genre_dummies = pd.DataFrame(mlb.fit_transform(df['genres_list']),
                                 columns=[f"genre__{g}" for g in mlb.classes_],
                                 index=df.index)
    # Unir dummies y quitar la columna original
    df = pd.concat([df, genre_dummies], axis=1)
    df.drop(columns=['genres','genres_list'], inplace=True, errors='ignore')

    # 3) Actores: extraer N primeros actores (5)
    def extract_top_n_actors(cast_str, n=extract_actors_n):
        parts = safe_split_first(cast_str)
        actors = parts[:n]
        while len(actors) < n:
            actors.append(None)
        return actors

    actor_cols = [f"actor_{i+1}" for i in range(extract_actors_n)]
    actors_df = df['cast'].apply(lambda s: pd.Series(extract_top_n_actors(s, extract_actors_n), index=actor_cols))
    df = pd.concat([df, actors_df], axis=1)
    df.drop(columns=['cast'], inplace=True, errors='ignore')

    # 4) Director principal (primer nombre si hay varios separados por coma)
    df['director'] = df['directors'].apply(lambda s: safe_split_first(s)[0] if len(safe_split_first(s))>0 else None)
    df.drop(columns=['directors'], inplace=True, errors='ignore')

    # 5) production_countries / spoken_languages -> primer valor (opcional)
    df['production_country_main'] = df.get('production_countries', pd.Series()).apply(lambda s: safe_split_first(s)[0] if len(safe_split_first(s))>0 else None)
    df['spoken_language_main'] = df.get('spoken_languages', pd.Series()).apply(lambda s: safe_split_first(s)[0] if len(safe_split_first(s))>0 else None)

    # 6) adult -> convertir a 0/1 si existe
    if 'adult' in df.columns:
        df['adult'] = df['adult'].astype(str).map(lambda x: 1 if x.lower() in ['true','1','t','yes'] else 0)

    # 7) Reducción de cardinalidad en actores/director (por defecto min_freq)
    #    Si preferís mantener todos los nombres, comentar las líneas siguientes.
    for col in actor_cols + ['director']:
        if col in df.columns:
            df[col] = top_k_or_minfreq_replace(df[col], min_freq=min_freq)

    # 8) Selección final de features (ajustá según lo que quieras usar)
    feature_cols = [
        # numéricas
        'runtime', 'popularity', 'vote_count', 'numVotes', 'release_year', 'release_month', 'adult',
        # categóricas
        'original_language','status','production_country_main','spoken_language_main'
    ]
    # añadir actores y director si existen
    feature_cols += [c for c in actor_cols if c in df.columns] + (['director'] if 'director' in df.columns else [])

    # añadir columnas de género (dummies) - ya son columnas binarias 0/1
    genre_cols = [c for c in df.columns if c.startswith('genre__')]
    feature_cols += genre_cols

    # filtrar columnas que realmente existan
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col = 'vote_average'  # ajustá si tu target tiene otro nombre

    result_df = df[feature_cols + [target_col]].copy()
    return result_df, feature_cols, target_col, genre_cols

# ---------------------------
# Preprocessor (scikit-learn)
# ---------------------------
def build_preprocessor(num_features, cat_features, genre_features):
    numeric_transformer = StandardScaler()
    # Para OneHotEncoder: si tenés muchísimas categorías, pon sparse=True para ahorrar memoria
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # ColumnTransformer: 
    # - num: escalado
    # - cat: OHE para actores/director/idioma/status
    # - genre: passthrough (ya son dummies 0/1)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features),
            ('genres', 'passthrough', genre_features)
        ],
        remainder='drop'
    )
    return preprocessor

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"No existe {INPUT_CSV} en el directorio actual.")

    print("Cargando CSV...")
    df_raw = pd.read_csv(INPUT_CSV)
    print("Dimensiones originales:", df_raw.shape)

    print("Aplicando feature engineering (genres con MultiLabelBinarizer, 5 actores)...")
    df_fe, feature_cols, target_col, genre_cols = engineer_features(df_raw, min_freq=MIN_FREQ, extract_actors_n=5)

    print("\nColumnas finales seleccionadas como features:")
    print(feature_cols)
    print("\nPrimeras filas:")
    print(df_fe.head())

    # Guardar CSV procesado
    df_fe.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataframe con features guardado en: {OUTPUT_CSV}")

    # Separar num y cat (para el preprocessor)
    # num: columnas numéricas reales
    num_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_fe[c])]
    # cat: las que no sean numéricas ni las columnas de género (ya binarias)
    cat_features = [c for c in feature_cols if c not in num_features and c not in genre_cols]

    print("\nNum features:", num_features)
    print("Cat features:", cat_features)
    print("Genre features (binarias):", genre_cols)

    preprocessor = build_preprocessor(num_features, cat_features, genre_cols)

    # Fit del preprocessor con X
    print("\nFiteando preprocessor con las features (puede tardar un poco)...")
    X = df_fe[feature_cols]
    preprocessor.fit(X)
    print("Preprocessor fiteado.")

    # Guardar preprocessor
    joblib.dump({
        'preprocessor': preprocessor,
        'num_features': num_features,
        'cat_features': cat_features,
        'genre_features': genre_cols,
        'feature_cols': feature_cols,
        'target_col': target_col
    }, PREPROCESSOR_PATH)
    print(f"Preprocessor y metadatos guardados en: {PREPROCESSOR_PATH}")

    print("\nListo. Próximo paso: pipeline (preprocessor + modelo).")
