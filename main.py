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
    print("🎨 Tema visual configurado.")

def cargar_datos():
    """Descarga y carga el dataset en un DataFrame de Pandas."""
    try:
        ruta_csv = descargar_dataset()
        df = pd.read_csv(ruta_csv)
        print("\nDataset cargado exitosamente. Primeras 5 filas:")
        print(df.head())
        return df
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")
        return None

# ================================
# 3. ANÁLISIS DE VALORES FALTANTES (DATASET ORIGINAL)
# ================================
def analizar_faltantes(df):
    """Analiza y visualiza los valores faltantes, tratando los ceros como nulos en columnas específicas."""
    print("\n--- 3. ANÁLISIS DE VALORES FALTANTES (DATASET ORIGINAL) ---")
    df_nulos = df.copy()
    
    cols_ceros_como_nulos = ["revenue", "budget", "runtime", "averageRating", "numVotes"]
    df_nulos[cols_ceros_como_nulos] = df_nulos[cols_ceros_como_nulos].replace(0, np.nan)
    
    faltantes_df = df_nulos.isnull().sum().reset_index()
    faltantes_df.columns = ["Columna", "Cantidad de Nulos"]
    faltantes_df["Porcentaje"] = (faltantes_df["Cantidad de Nulos"] / len(df_nulos)) * 100
    
    con_faltantes = faltantes_df[faltantes_df["Cantidad de Nulos"] > 0].sort_values(by="Porcentaje", ascending=False)
    sin_faltantes = faltantes_df[faltantes_df["Cantidad de Nulos"] == 0]
    
    print("\nColumnas CON valores faltantes:")
    print(con_faltantes)
    
    print("\nColumnas SIN valores faltantes:")
    print(sin_faltantes["Columna"].tolist())
    
    cols_con_nulos = con_faltantes['Columna'].tolist()
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_nulos[cols_con_nulos].isnull(), cbar=False, cmap="viridis_r")
    plt.title("Mapa de Calor de Valores Faltantes (Dataset Original)", fontsize=20, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.show()
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
        print("Filas duplicadas eliminadas.")
        return df_sin_duplicados
    return df

# ================================
# 5. ANÁLISIS DE OUTLIERS
# ================================
def analizar_outliers(df, columna):
    """Detecta y visualiza outliers para una columna numérica específica."""
    print(f"\n--- 5. ANÁLISIS DE OUTLIERS EN '{columna}' ---")
    if pd.api.types.is_numeric_dtype(df[columna]):
        df_col = df.dropna(subset=[columna])
        Q1, Q3 = df_col[columna].quantile(0.25), df_col[columna].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = df_col[(df_col[columna] < lim_inf) | (df_col[columna] > lim_sup)]
        print(f"Límites para outliers: Inferior < {lim_inf:.2f}, Superior > {lim_sup:.2f}")
        print(f"Número de outliers detectados: {outliers.shape[0]}")
        limite_visual = df_col[columna].quantile(0.995)
        df_visual = df_col[df_col[columna] < limite_visual]
        plt.figure(figsize=(16, 4))
        sns.boxplot(x=df_visual[columna])
        plt.title(f"Diagrama de Caja para '{columna}' (Visualización del 99.5% de los datos)", fontsize=18, weight='bold')
        plt.xlabel(f"{columna} (filtrado para visualización)", fontsize=14)
        plt.show()

# ================================
# 6. ANÁLISIS DE CORRELACIÓN
# ================================
def analizar_correlacion(df):
    """Calcula y visualiza la matriz de correlación de las variables numéricas."""
    print("\n--- 6. ANÁLISIS DE CORRELACIÓN ---")
    num_df = df.select_dtypes(include=np.number).drop(columns=['id'], errors='ignore')
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # ## PERFECCIONAMIENTO FINAL DEL GRÁFICO ##
    # Ajustar dinámicamente el tamaño de la figura para evitar que las etiquetas se corten
    num_vars = len(corr.columns)
    fig_size = max(12, num_vars) # Establecer un tamaño mínimo de 12x12
    plt.figure(figsize=(fig_size, fig_size))
    
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        linewidths=.5, 
        annot_kws={"size": 10}, # Tamaño de fuente consistente
        square=True
    )
    plt.title("Matriz de Correlación de Variables Numéricas", fontsize=20, weight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout(pad=1.0) # Usar un padding menor para evitar cortes
    plt.show()

# ================================
# 7. VISUALIZACIONES EXPLORATORIAS
# ================================
def generar_visualizaciones(df):
    """Genera una serie de gráficos para el análisis exploratorio de datos."""
    print("\n--- 7. VISUALIZACIONES EXPLORATORIAS ---")
    # Histograma
    plt.figure()
    sns.histplot(df["vote_average"].dropna(), kde=True, bins=30)
    plt.title("Distribución del Promedio de Votos (vote_average)", fontsize=18, weight='bold')
    plt.tight_layout()
    plt.show()
    
    # Géneros
    if "genres" in df.columns:
        print("\nProcesando y graficando géneros...")
        df_genres = df.dropna(subset=['genres']).copy()
        df_genres['genres_list'] = df_genres['genres'].str.split(', ')
        df_exploded = df_genres.explode('genres_list').reset_index(drop=True)
        df_exploded['genres_list'] = df_exploded['genres_list'].str.strip()
        plt.figure(figsize=(14, 10))
        sns.countplot(y=df_exploded['genres_list'], order=df_exploded['genres_list'].value_counts().index[:15], palette='viridis', hue=df_exploded['genres_list'], legend=False)
        plt.title("Top 15 Géneros de Películas", fontsize=18, weight='bold')
        plt.tight_layout()
        plt.show()
        
    # Evolución temporal
    if "release_date" in df.columns:
        print("\nAnalizando evolución temporal por año de estreno...")
        df_temp = df.copy()
        df_temp["release_date"] = pd.to_datetime(df_temp["release_date"], errors="coerce")
        df_temp["release_year"] = df_temp["release_date"].dt.year
        df_reciente = df_temp[df_temp['release_year'] >= 1980]
        plt.figure(figsize=(20, 8))
        ax = sns.countplot(x=df_reciente["release_year"], color=sns.color_palette("muted")[2])
        ax.set_title("Cantidad de Películas Estrenadas por Año (desde 1980)", fontsize=18, weight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# ================================
# 8. CREACIÓN DEL DATASET REDUCIDO
# ================================
def crear_dataset_reducido(df):
    """Limpia el dataset, guarda una versión reducida y devuelve la ruta del archivo guardado."""
    print("\n--- 8. CREACIÓN DEL DATASET REDUCIDO ---")
    df_limpio = df.copy()
    
    cols_ceros_nan = ["revenue", "budget", "runtime", "averageRating", "numVotes"]
    df_limpio[cols_ceros_nan] = df_limpio[cols_ceros_nan].replace(0, np.nan)

    cols_a_eliminar = ["budget", "revenue", "tagline", "poster_path", "backdrop_path", "homepage"]
    print(f"\nColumnas a eliminar: {cols_a_eliminar}")
    df_limpio = df_limpio.drop(columns=cols_a_eliminar, errors="ignore")
    print("Columnas eliminadas.")
    
    cols_clave_nulos = ["vote_average", "vote_count", "release_date", "genres", "runtime"]
    df_limpio = df_limpio.dropna(subset=cols_clave_nulos)
    print(f"Filas restantes tras eliminar nulos en columnas clave: {df_limpio.shape[0]}")
    
    Q1, Q3 = df_limpio["vote_average"].quantile(0.25), df_limpio["vote_average"].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df_final = df_limpio[(df_limpio["vote_average"] >= lim_inf) & (df_limpio["vote_average"] <= lim_sup)]
    print(f"Filas restantes tras limpiar outliers en 'vote_average': {df_final.shape[0]}")

    if 10000 <= df_final.shape[0] <= 20000:
        df_reducido = df_final
    elif df_final.shape[0] > 20000:
        df_reducido = df_final.sample(n=20000, random_state=42)
    else:
        df_reducido = df_final
        print(f"⚠️ Advertencia: El dataset final tiene {df_final.shape[0]} registros, menos de los 10,000 esperados.")

    ruta_salida = "movies_dataset_reducido.csv"
    df_reducido.to_csv(ruta_salida, index=False)
    print(f"\n✅ Dataset reducido con {df_reducido.shape[0]} registros guardado en: '{ruta_salida}'")
    
    return ruta_salida

# ================================
# 9. ANÁLISIS DEL DATASET REDUCIDO
# ================================
def analizar_dataset_reducido(ruta_csv_reducido):
    """Carga y analiza el dataset reducido para verificar su estado final."""
    print("\n--- 9. ANÁLISIS DEL DATASET REDUCIDO ---")
    try:
        df_reducido = pd.read_csv(ruta_csv_reducido)
        print("Dataset reducido cargado para verificación.")
        
        faltantes_reducido = df_reducido.isnull().sum().reset_index()
        faltantes_reducido.columns = ["Columna", "Cantidad de Nulos"]
        
        con_faltantes = faltantes_reducido[faltantes_reducido["Cantidad de Nulos"] > 0]
        # ## NUEVA FUNCIONALIDAD: Mostrar columnas sin nulos en el dataset reducido ##
        sin_faltantes = faltantes_reducido[faltantes_reducido["Cantidad de Nulos"] == 0]

        if con_faltantes.empty:
            print("\n✅ ¡Excelente! No se detectaron valores faltantes en el dataset reducido.")
        else:
            print("\n⚠️ Se detectaron valores faltantes en el dataset reducido:")
            print(con_faltantes.sort_values(by="Cantidad de Nulos", ascending=False))
            
            plt.figure(figsize=(16, 8))
            sns.heatmap(df_reducido[con_faltantes['Columna'].tolist()].isnull(), cbar=False, cmap="plasma")
            plt.title("Mapa de Valores Faltantes (Dataset Reducido)", fontsize=20, weight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.tight_layout()
            plt.show()

        if not sin_faltantes.empty:
            print("\nColumnas SIN valores faltantes en el dataset reducido:")
            print(sin_faltantes["Columna"].tolist())

        print("\nColumnas finales en el dataset reducido:")
        print(df_reducido.columns.tolist())
        
        print("\nPrimeras filas del dataset reducido:")
        print(df_reducido.head())

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo del dataset reducido en la ruta '{ruta_csv_reducido}'")

# ================================
# SCRIPT PRINCIPAL
# ================================
if __name__ == "__main__":
    configurar_visualizaciones()
    df_original = cargar_datos()
    
    if df_original is not None:
        df_con_nulos = analizar_faltantes(df_original)
        df_sin_duplicados = analizar_duplicados(df_con_nulos)
        
        analizar_outliers(df_sin_duplicados, "vote_average")
        analizar_outliers(df_sin_duplicados, "runtime")
        
        analizar_correlacion(df_sin_duplicados)
        generar_visualizaciones(df_sin_duplicados)
        
        ruta_reducido_final = crear_dataset_reducido(df_sin_duplicados)
        if ruta_reducido_final:
            analizar_dataset_reducido(ruta_reducido_final)