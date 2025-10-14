# ================================
# 1. LIBRERÍAS
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from descargar_dataset import descargar_dataset

# ================================
# 2. CONFIGURACIÓN Y CARGA DE DATOS
# ================================
def configurar_visualizaciones():
    """Establece un tema visual consistente para todos los gráficos."""
    sns.set_theme(style="whitegrid", palette="muted", context="talk")
    plt.rcParams['figure.figsize'] = (14, 8)
    print("✔ Tema visual configurado.")

def cargar_datos():
    """Descarga y carga el dataset en un DataFrame de Pandas."""
    try:
        ruta_csv = descargar_dataset()
        df = pd.read_csv(ruta_csv)
        print(f"\n✔ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas.")
        print(df.head())
        return df
    except Exception as e:
        print(f"✖ Error al cargar los datos: {e}")
        return None

# ================================
# 3. ANÁLISIS DE VALORES FALTANTES (DATASET ORIGINAL)
# ================================
def analizar_faltantes(df):
    """Analiza y visualiza los valores faltantes, tratando los ceros como nulos en columnas específicas."""
    print("\n--- 3. ANÁLISIS DE VALORES FALTANTES (DATASET ORIGINAL) ---")
    df_nulos = df.copy()

    cols_ceros_como_nulos = ["revenue", "budget", "runtime", "averageRating",
                             "numVotes", "vote_average", "vote_count"]
    cols_presentes = [c for c in cols_ceros_como_nulos if c in df_nulos.columns]
    if cols_presentes:
        df_nulos[cols_presentes] = df_nulos[cols_presentes].replace(0, np.nan)

    # Tabla de faltantes
    faltantes_df = (
        df_nulos.isnull()
        .sum()
        .reset_index(name="Cantidad de Nulos")
        .rename(columns={"index": "Columna"})
    )
    faltantes_df["Porcentaje"] = (faltantes_df["Cantidad de Nulos"] / len(df_nulos)) * 100

    con_faltantes = faltantes_df[faltantes_df["Cantidad de Nulos"] > 0] \
        .sort_values(by="Porcentaje", ascending=False)
    sin_faltantes = faltantes_df[faltantes_df["Cantidad de Nulos"] == 0]

    print("\nColumnas CON valores faltantes (top 10):")
    print(con_faltantes.head(10))
    print("\nColumnas SIN valores faltantes (cantidad):", sin_faltantes.shape[0])

    # Heatmap de nulos (si corresponde)
    cols_con_nulos = con_faltantes['Columna'].tolist()
    if cols_con_nulos:
        plt.figure(figsize=(min(20, 8 + 0.7*len(cols_con_nulos)), 10))
        sns.heatmap(df_nulos[cols_con_nulos].isnull(), cbar=False, cmap="viridis_r")
        plt.title("Mapa de Calor de Valores Faltantes (Dataset Original)", fontsize=20, weight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.show()

        # Barras Top-20 por % de nulos con etiquetas
        top = con_faltantes.head(20)
        plt.figure(figsize=(12, max(6, int(0.4*len(top)))))
        ax = sns.barplot(data=top, x="Porcentaje", y="Columna", orient="h")
        for i, v in enumerate(top["Porcentaje"]):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
        plt.title("Top 20 columnas por % de nulos", fontsize=18, weight="bold")
        plt.xlabel("% de nulos")
        plt.ylabel("Columna")
        plt.tight_layout()
        plt.show()
    else:
        print("✔ No hay columnas con valores faltantes.")

    return df_nulos

# ================================
# 4. ANÁLISIS DE DUPLICADOS
# ================================
def analizar_duplicados(df):
    """Encuentra y elimina filas duplicadas."""
    print("\n--- 4. ANÁLISIS DE DUPLICADOS ---")
    duplicados = df.duplicated().sum()
    print(f"Número de filas duplicadas encontradas: {duplicados}")
    if duplicados > 0:
        df_sin_duplicados = df.drop_duplicates(keep='first')
        print("✔ Filas duplicadas eliminadas.")
        return df_sin_duplicados
    return df

# ================================
# 5. ANÁLISIS DE OUTLIERS
# ================================
def analizar_outliers(df, columna):
    """Detecta y visualiza outliers para una columna numérica específica."""
    print(f"\n--- 5. ANÁLISIS DE OUTLIERS EN '{columna}' ---")
    if columna not in df.columns or not pd.api.types.is_numeric_dtype(df[columna]):
        print("✖ Columna no numérica o no existe. Se omite.")
        return

    df_col = df.dropna(subset=[columna])
    Q1, Q3 = df_col[columna].quantile(0.25), df_col[columna].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df_col[(df_col[columna] < lim_inf) | (df_col[columna] > lim_sup)]
    print(f"Límites para outliers: LI={lim_inf:.2f}, Q1={Q1:.2f}, Q3={Q3:.2f}, LS={lim_sup:.2f}")
    print(f"Número de outliers detectados: {outliers.shape[0]}")

    # Recorte para visual (99.5%) para evitar colas extremas
    limite_visual = df_col[columna].quantile(0.995)
    df_visual = df_col[df_col[columna] <= limite_visual]

    plt.figure(figsize=(16, 4))
    ax = sns.boxplot(x=df_visual[columna], showfliers=False)
    # Líneas guía
    for x, ls, lbl in [(lim_inf, ":", "LI"), (Q1, "--", "Q1"), (Q3, "--", "Q3"), (lim_sup, ":", "LS")]:
        ax.axvline(x, linestyle=ls)
        ax.text(x, 0.02, lbl, transform=ax.get_xaxis_transform(), ha="center")
    plt.title(f"Diagrama de Caja '{columna}' (99.5% visible) | Outliers={outliers.shape[0]}", fontsize=18, weight='bold')
    plt.xlabel(f"{columna}")
    plt.tight_layout()
    plt.show()

# ================================
# 6. ANÁLISIS DE CORRELACIÓN
# ================================
def analizar_correlacion(df):
    """Calcula y visualiza la matriz de correlación de las variables numéricas."""
    print("\n--- 6. ANÁLISIS DE CORRELACIÓN ---")
    num_df = df.select_dtypes(include=np.number).drop(columns=['id'], errors='ignore')
    if num_df.empty or num_df.shape[1] < 2:
        print("✖ Muy pocas variables numéricas para correlación.")
        return

    # Evitar columnas constantes
    std = num_df.std(numeric_only=True)
    num_df = num_df.loc[:, std > 0]
    if num_df.shape[1] < 2:
        print("✖ Tras remover constantes, no hay suficientes variables.")
        return

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Ajuste dinámico del tamaño
    num_vars = len(corr.columns)
    fig_size = max(12, min(24, int(num_vars * 0.8)))
    plt.figure(figsize=(fig_size, fig_size))

    annot_flag = num_vars <= 14
    sns.heatmap(
        corr, mask=mask, annot=annot_flag, cmap="coolwarm", fmt=".2f",
        linewidths=.5, square=True, cbar=True, annot_kws={"size": 10}
    )
    plt.title("Matriz de Correlación de Variables Numéricas", fontsize=20, weight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout(pad=1.0)
    plt.show()

# ================================
# 7. VISUALIZACIONES EXPLORATORIAS
# ================================
def generar_visualizaciones(df):
    """Genera una serie de gráficos para el análisis exploratorio de datos."""
    print("\n--- 7. VISUALIZACIONES EXPLORATORIAS ---")

    # 7.1 Histograma vote_average con media/mediana
    if "vote_average" in df.columns:
        s = pd.to_numeric(df["vote_average"], errors="coerce").dropna()
        plt.figure()
        ax = sns.histplot(s, kde=True, bins=30)
        med, mean = s.median(), s.mean()
        ax.axvline(med, linestyle="--", label=f"Mediana {med:.2f}")
        ax.axvline(mean, linestyle=":", label=f"Media {mean:.2f}")
        plt.legend()
        plt.title("Distribución de vote_average (con media y mediana)", fontsize=18, weight='bold')
        plt.xlabel("vote_average")
        plt.tight_layout()
        plt.show()

    # 7.2 Histograma runtime (recortado al 99%) con media/mediana
    if "runtime" in df.columns:
        r = pd.to_numeric(df["runtime"], errors="coerce").dropna()
        if not r.empty:
            upper = r.quantile(0.99)
            r_clip = r.clip(upper=upper)
            plt.figure()
            ax = sns.histplot(r_clip, kde=True, bins=40)
            med, mean = r.median(), r.mean()
            ax.axvline(med, linestyle="--", label=f"Mediana {med:.0f} min")
            ax.axvline(mean, linestyle=":", label=f"Media {mean:.0f} min")
            plt.legend()
            plt.title("Distribución de runtime (min) — recorte 99% para legibilidad", fontsize=18, weight='bold')
            plt.xlabel("runtime (minutos)")
            plt.tight_layout()
            plt.show()

    # 7.3 Top-15 géneros (en porcentaje, con etiquetas)
    if "genres" in df.columns:
        print("\nProcesando y graficando géneros...")
        s = df["genres"].dropna().astype(str)
        sep = "|" if s.str.contains(r"\|").any() else ","
        exploded = s.str.split(sep).explode().str.strip()
        exploded = exploded[exploded.ne("").fillna(False)]
        counts = exploded.value_counts().head(15)
        total = exploded.shape[0]
        top_df = pd.DataFrame({
            "Género": counts.index,
            "Porcentaje": (counts.values / total) * 100
        })
        plt.figure(figsize=(14, 10))
        ax = sns.barplot(data=top_df, x="Porcentaje", y="Género", orient='h')
        for i, v in enumerate(top_df["Porcentaje"]):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
        plt.title("Top 15 Géneros de Películas (en % del total)", fontsize=18, weight='bold')
        plt.xlabel("% del total de menciones de géneros")
        plt.ylabel("Género")
        plt.tight_layout()
        plt.show()

    # 7.4 Evolución temporal (conteo anual + media móvil 5 años)
    if "release_date" in df.columns:
        print("\nAnalizando evolución temporal por año de estreno...")
        df_temp = df.copy()
        df_temp["release_date"] = pd.to_datetime(df_temp["release_date"], errors="coerce")
        df_temp["release_year"] = df_temp["release_date"].dt.year
        counts = df_temp["release_year"].value_counts().sort_index()
        counts = counts[counts.index >= 1980]
        if not counts.empty:
            roll = counts.rolling(window=5, min_periods=1).mean()
            plt.figure(figsize=(20, 8))
            sns.lineplot(x=counts.index, y=counts.values, label="Conteo anual")
            sns.lineplot(x=roll.index, y=roll.values, label="Media móvil (5 años)")
            plt.title("Películas estrenadas por año (≥ 1980) con tendencia 5 años", fontsize=18, weight='bold')
            plt.xlabel("Año de estreno")
            plt.ylabel("Cantidad")
            plt.legend()
            plt.tight_layout()
            plt.show()

# ================================
# 8. CREACIÓN DEL DATASET REDUCIDO
# ================================
def crear_dataset_reducido(df):
    """Limpia el dataset, guarda una versión reducida y devuelve la ruta del archivo guardado."""
    print("\n--- 8. CREACIÓN DEL DATASET REDUCIDO ---")
    df_limpio = df.copy()

    cols_ceros_nan = ["revenue", "budget", "vote_average", "vote_count",
                      "runtime", "averageRating", "numVotes"]
    cols_ceros_presentes = [c for c in cols_ceros_nan if c in df_limpio.columns]
    if cols_ceros_presentes:
        df_limpio[cols_ceros_presentes] = df_limpio[cols_ceros_presentes].replace(0, np.nan)

    cols_a_eliminar = ["budget", "revenue", "tagline", "poster_path", "backdrop_path", "homepage"]
    print(f"\nColumnas a eliminar: {cols_a_eliminar}")
    df_limpio = df_limpio.drop(columns=cols_a_eliminar, errors="ignore")
    print("✔ Columnas eliminadas.")

    cols_clave_nulos = ["vote_average", "vote_count", "release_date", "genres",
                        "production_countries", "overview", "production_companies",
                        "spoken_languages", "keywords", "directors", "writers", "cast", "runtime"]
    cols_clave_presentes = [c for c in cols_clave_nulos if c in df_limpio.columns]
    df_limpio = df_limpio.dropna(subset=cols_clave_presentes)
    print(f"✔ Filas restantes tras eliminar nulos en columnas clave: {df_limpio.shape[0]}")

    # Filtrado de outliers por IQR en vote_average (igual que tu lógica original)
    if "vote_average" in df_limpio.columns:
        Q1, Q3 = df_limpio["vote_average"].quantile(0.25), df_limpio["vote_average"].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df_final = df_limpio[(df_limpio["vote_average"] >= lim_inf) & (df_limpio["vote_average"] <= lim_sup)]
        print(f"✔ Filas tras limpiar outliers en 'vote_average': {df_final.shape[0]}")
    else:
        df_final = df_limpio

    # Ajuste a rango 10k–20k
    if 10000 <= df_final.shape[0] <= 20000:
        df_reducido = df_final
    elif df_final.shape[0] > 20000:
        df_reducido = df_final.sample(n=20000, random_state=42)
    else:
        df_reducido = df_final
        print(f"⚠ Advertencia: El dataset final tiene {df_final.shape[0]} registros, menos de los 10,000 esperados.")

    ruta_salida = "movies_dataset_reducido.csv"
    df_reducido.to_csv(ruta_salida, index=False)
    print(f"\n✔ Dataset reducido con {df_reducido.shape[0]} registros guardado en: '{ruta_salida}'")

    return ruta_salida

# ================================
# 9. ANÁLISIS DEL DATASET REDUCIDO
# ================================
def analizar_dataset_reducido(ruta_csv_reducido):
    """Carga y analiza el dataset reducido para verificar su estado final."""
    print("\n--- 9. ANÁLISIS DEL DATASET REDUCIDO ---")
    try:
        df_reducido = pd.read_csv(ruta_csv_reducido)
        print(f"✔ Dataset reducido cargado para verificación: {df_reducido.shape}")

        faltantes_reducido = (
            df_reducido.isnull()
            .sum()
            .reset_index(name="Cantidad de Nulos")
            .rename(columns={"index": "Columna"})
        )

        con_faltantes = faltantes_reducido[faltantes_reducido["Cantidad de Nulos"] > 0]
        sin_faltantes = faltantes_reducido[faltantes_reducido["Cantidad de Nulos"] == 0]

        if con_faltantes.empty:
            print("\n✔ No se detectaron valores faltantes en el dataset reducido.")
        else:
            print("\n⚠ Se detectaron valores faltantes en el dataset reducido:")
            print(con_faltantes.sort_values(by="Cantidad de Nulos", ascending=False))

            plt.figure(figsize=(16, 8))
            sns.heatmap(df_reducido[con_faltantes['Columna'].tolist()].isnull(), cbar=False, cmap="plasma")
            plt.title("Mapa de Valores Faltantes (Dataset Reducido)", fontsize=20, weight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.tight_layout()
            plt.show()

        print("\nColumnas finales en el dataset reducido:")
        print(df_reducido.columns.tolist())

        print("\nPrimeras filas del dataset reducido:")
        print(df_reducido.head())

    except FileNotFoundError:
        print(f"✖ Error: No se encontró el archivo del dataset reducido en la ruta '{ruta_csv_reducido}'")

# ================================
# SCRIPT PRINCIPAL
# ================================
if __name__ == "__main__":
    configurar_visualizaciones()
    df_original = cargar_datos()

    if df_original is not None:
        # 3) Faltantes (heatmap + barras %)
        df_con_nulos = analizar_faltantes(df_original)

        # 4) Duplicados
        df_sin_duplicados = analizar_duplicados(df_con_nulos)

        # 5) Outliers (diagnóstico visual + límites anotados)
        analizar_outliers(df_sin_duplicados, "vote_average")
        analizar_outliers(df_sin_duplicados, "runtime")

        # 6) Correlación
        analizar_correlacion(df_sin_duplicados)

        # 7) Visualizaciones clave (voto, runtime, géneros %, tiempo + tendencia)
        generar_visualizaciones(df_sin_duplicados)

        # 8) Dataset reducido y 9) Verificación final
        ruta_reducido_final = crear_dataset_reducido(df_sin_duplicados)
        if ruta_reducido_final:
            analizar_dataset_reducido(ruta_reducido_final)
