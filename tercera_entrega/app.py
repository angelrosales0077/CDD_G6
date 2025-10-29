import streamlit as st
import joblib
import pandas as pd
import os
import re

st.title("🎬 Predicción de Calificación de Películas")

# Intentar cargar dataset para poblar dropdowns (fallback a valores por defecto)
DATA_CSV = "movies_dataset_fe_mlb_5actors.csv"
lang_options = ["en", "es", "fr"]
actor_options = []
director_options = []
if os.path.exists(DATA_CSV):
    try:
        df_all = pd.read_csv(DATA_CSV, low_memory=False)
        if 'original_language' in df_all.columns:
            lang_options = sorted(df_all['original_language'].dropna().astype(str).unique().tolist())
        # buscar columnas actor_1..actor_5
        actor_cols = [c for c in df_all.columns if c.startswith("actor_")]
        if actor_cols:
            actors = pd.unique(df_all[actor_cols].fillna("").values.ravel())
            actor_options = sorted([a for a in actors.astype(str) if a.strip() != ""])
        # top directors
        if 'director' in df_all.columns:
            director_options = sorted(df_all['director'].dropna().astype(str).unique().tolist())
    except Exception:
        pass

# Entradas básicas (ahora dropdowns para idioma y actores)
runtime = st.slider("Duración (minutos)", 60, 240, 120)
popularity = st.number_input("Popularidad", value=50.0)
vote_count = st.number_input("Votos", value=1000)
release_year = st.number_input("Año de estreno", 1950, 2025, 2023)
release_month = st.slider("Mes", 1, 12, 7)

# Idioma como selectbox (fallback a text_input si no hay opciones)
if lang_options:
    language = st.selectbox("Idioma original", options=lang_options, index=lang_options.index("en") if "en" in lang_options else 0)
else:
    language = st.text_input("Idioma original", "en")

# Actores como multiselect (máx 5 seleccionables)
if actor_options:
    selected_actors = st.multiselect("Actores principales (max 5)", options=actor_options)
    if len(selected_actors) > 5:
        st.warning("Has seleccionado más de 5 actores; se usarán los primeros 5.")
        selected_actors = selected_actors[:5]
else:
    # Ya no permitimos entrada manual: informar y dejar lista vacía
    st.info("No hay lista de actores en el dataset — se dejará vacío (sin actores).")
    selected_actors = []

# Director: selectbox si hay opciones, sino dejar vacío (sin input manual)
if director_options:
    director_input = st.selectbox("Director (opcional)", options=[""] + director_options)
else:
    st.info("No hay lista de directores en el dataset — se dejará vacío (sin director).")
    director_input = ""

# Candidatos comunes de preprocessor
preproc_candidates = [
    "models_saved/preprocessor.joblib",
    "preprocessor_mlb_5actors.joblib",
    "preprocessor.joblib",
    "preprocessor_mlb_5actors.joblib"
]

# Cargar modelo a predecir (si existe)
model_path = "models_saved/xgboost.joblib"
if not os.path.exists(model_path):
    st.error(f"No se encontró el modelo en {model_path}")
    st.stop()

# Cargar preprocessor y feature_cols si existen
preprocessor = None
feature_cols = None
for p in preproc_candidates:
    if os.path.exists(p):
        loaded = joblib.load(p)
        if isinstance(loaded, dict) and 'preprocessor' in loaded:
            preprocessor = loaded['preprocessor']
            feature_cols = loaded.get('feature_cols', None)
        else:
            preprocessor = loaded
        break

if preprocessor is None:
    st.error(f"No se encontró el preprocessor. Buscado en: {', '.join(preproc_candidates)}")
    st.stop()

# Determinar columnas esperadas por el preprocessor (preferir feature_cols guardadas)
expected_cols = None
if feature_cols is not None:
    expected_cols = feature_cols
else:
    try:
        if hasattr(preprocessor, "feature_names_in_"):
            expected_cols = list(preprocessor.feature_names_in_)
    except Exception:
        expected_cols = None

# Obtener lista de géneros disponibles (si hay columnas genre__*)
genre_cols = []
if expected_cols is not None:
    genre_cols = [c for c in expected_cols if c.startswith("genre__")]
# lista de nombres legibles
genre_names = [c.split("__", 1)[1] for c in genre_cols] if genre_cols else ["Action", "Drama", "Comedy", "Horror", "Sci-Fi"]

# UI: géneros
selected_genres = st.multiselect("Géneros", options=sorted(genre_names), default=[])

# ----- INPUTS MANUALES ELIMINADOS -----
# Ya no se muestran los st.text_input para 'Actor #i'
# Ya no se muestra el st.text_input para 'Director'


# Botón predict
if st.button("Predecir"):
    model = joblib.load(model_path)

    # Construir DataFrame base con valores provistos por el usuario
    base = {
        "runtime": runtime,
        "popularity": popularity,
        "vote_count": vote_count,
        "numVotes": vote_count,
        "release_year": release_year,
        "release_month": release_month,
        "adult": False,
        "original_language": language,
        "status": "Released",
        "production_country_main": "United States of America",
        "spoken_language_main": "English",
    }
    X_raw = pd.DataFrame([base])

    def _normalize_key(s):
        return re.sub(r'\W+', '', str(s).lower())

    # Si conocemos expected_cols, crear DataFrame alineado y mapear géneros y actores
    if expected_cols is not None:
        X_pre = pd.DataFrame(columns=expected_cols)
        # rellenar columnas con valores por defecto razonables
        for c in expected_cols:
            if c in X_raw.columns:
                X_pre.at[0, c] = X_raw.at[0, c]
            elif c.startswith("genre__") or c.startswith("num_") or c in ["vote_count", "numVotes", "popularity", "runtime"]:
                X_pre.at[0, c] = 0
            elif c.startswith("actor_") or c in ["director","original_language","status","production_country_main","spoken_language_main"]:
                X_pre.at[0, c] = ""
            elif c in ["release_year","release_month"]:
                X_pre.at[0, c] = X_raw.at[0, c] if c in X_raw.columns else 0
            else:
                X_pre.at[0, c] = 0

        # Mapear géneros seleccionados a columnas genre__*
        # construir mapping normalized_genre_name -> genre_col
        genre_map = { _normalize_key(col.split("__",1)[1]) : col for col in genre_cols }
        for g in selected_genres:
            key = _normalize_key(g)
            if key in genre_map:
                X_pre.at[0, genre_map[key]] = 1
            else:
                # intento de coincidencia parcial (por ejemplo "sci-fi" vs "Science Fiction")
                for nm, colname in genre_map.items():
                    if key in nm or nm in key:
                        X_pre.at[0, colname] = 1
                        break

        # Rellenar actor_1..actor_5 y director si existen
        # (Usamos 'selected_actors' del multiselect en lugar de 'actor_inputs' manuales)
        for i, name in enumerate(selected_actors, start=1):
            col = f"actor_{i}"
            if col in expected_cols:
                X_pre.at[0, col] = name if name is not None else ""
        
        # (Usamos 'director_input' del selectbox)
        if "director" in expected_cols:
            X_pre.at[0, "director"] = director_input

    else:
        # fallback: usar X_raw y añadir géneros como columna 'genres' si el preprocessor espera parsearlas
        X_pre = X_raw.copy()
        X_pre["genres"] = selected_genres
        # añadir actors/director si el preprocessor podría leerlos
        # (Usamos 'selected_actors' del multiselect)
        X_pre["actors"] = selected_actors
        X_pre["director"] = director_input

    # Transformar y predecir (capturar errores para debug)
    try:
        X_trans = preprocessor.transform(X_pre)

        # Alinear columnas transformadas con las que espera el modelo
        def _get_preprocessor_output_names(prep, feature_cols=None):
            # intenta obtener nombres de salida del preprocessor (sklearn >=1.0)
            try:
                if feature_cols is not None:
                    return list(prep.get_feature_names_out(feature_cols))
                return list(prep.get_feature_names_out())
            except Exception:
                return None

        def _get_model_expected_names(model):
            # preferir sklearn attribute, luego xgboost booster names
            if hasattr(model, "feature_names_in_"):
                return list(model.feature_names_in_)
            try:
                booster = model.get_booster()
                if booster is not None and hasattr(booster, "feature_names") and booster.feature_names is not None:
                    return list(booster.feature_names)
            except Exception:
                pass
            return None

        out_names = _get_preprocessor_output_names(preprocessor, feature_cols=expected_cols)
        model_names = _get_model_expected_names(model)

        # Si tenemos nombres de columna, construir DataFrame y reindexar
        if out_names is not None and model_names is not None:
            df_out = pd.DataFrame(X_trans, columns=out_names)
            # Reindexar en el orden que el modelo espera, rellenando con 0 las columnas faltantes
            df_aligned = df_out.reindex(columns=model_names, fill_value=0)
            X_aligned = df_aligned.values
        else:
            # fallback por tamaño: recortar o rellenar con ceros
            n_expected = getattr(model, "n_features_in_", None)
            if n_expected is None:
                n_expected = model.get_booster().num_features() if hasattr(model, "get_booster") else None
            if n_expected is None:
                X_aligned = X_trans
            else:
                if X_trans.shape[1] > n_expected:
                    X_aligned = X_trans[:, :n_expected]
                elif X_trans.shape[1] < n_expected:
                    pad = np.zeros((X_trans.shape[0], n_expected - X_trans.shape[1]))
                    X_aligned = np.hstack([X_trans, pad])
                else:
                    X_aligned = X_trans

        # comprobar forma final
        if hasattr(model, "n_features_in_") and X_aligned.shape[1] != model.n_features_in_:
            raise ValueError(f"Feature shape mismatch after alignment, model expects {model.n_features_in_} features but got {X_aligned.shape[1]}")

        pred = model.predict(X_aligned)[0]
        st.success(f"Predicción esperada: {pred:.2f}")
    except Exception as e:
        st.error("Error durante la transformación o predicción.")
        st.write("Excepción:", e)
        st.write("DataFrame pasado al preprocessor (X_pre):")
        st.write(X_pre)