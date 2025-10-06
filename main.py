# ================================
# 1. LIBRERÍAS
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from descargar_dataset import descargar_dataset

# Configuración visual
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10,6)

# Descargar dataset y obtener ruta
ruta_csv = descargar_dataset()

# Cargar dataset
df = pd.read_csv(ruta_csv)
print("Dataset cargado. Primeras filas:")
print(df.head())

# ================================
# 3. ANÁLISIS DE VALORES FALTANTES
# ================================
df_nulos = df.copy()

cols_ceros_como_nulos = ["revenue", "vote_average", "vote_count", "budget"]
df_nulos[cols_ceros_como_nulos] = df_nulos[cols_ceros_como_nulos].replace(0, np.nan)

faltantes = df_nulos.isnull().sum().reset_index()
faltantes.columns = ["Columna", "Cantidad de Nulos"]
faltantes["Porcentaje"] = (faltantes["Cantidad de Nulos"] / len(df_nulos)) * 100
print(faltantes)

sns.heatmap(df_nulos.isnull(), cbar=False, cmap="viridis")
plt.title("Mapa de valores faltantes (considerando 0.0 como nulo)")
plt.show()

# ================================
# 4. ANÁLISIS DE DUPLICADOS
# ================================
duplicados = df.duplicated().sum()
print("Número de filas duplicadas:", duplicados)
df = df.drop_duplicates()

# ================================
# 5. ANÁLISIS DE OUTLIERS
# ================================
columna = "vote_average"  # Ahora se analiza sobre esta variable
df_no0 = df[df[columna] > 0]

Q1 = df_no0[columna].quantile(0.25)
Q3 = df_no0[columna].quantile(0.75)
IQR = Q3 - Q1
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR

outliers = df_no0[(df_no0[columna] < lim_inf) | (df_no0[columna] > lim_sup)]
print(f"Número de outliers en {columna}:", outliers.shape[0])

sns.boxplot(x=df_no0[columna])
plt.title(f"Detección de outliers en {columna}")
plt.show()

# ================================
# 6. ANÁLISIS DE CORRELACIÓN
# ================================
num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación")
plt.show()

# ================================
# 7. VISUALIZACIONES EXPLORATORIAS
# ================================

# Histograma de la variable numérica principal
sns.histplot(df["vote_average"], kde=True, bins=30)
plt.title("Distribución del promedio de votos (vote_average)")
plt.show()

# Gráfico de barras para variable categórica
sns.countplot(y=df["genres"], order=df["genres"].value_counts().index[:10])
plt.title("Frecuencia de géneros")
plt.show()

# Evolución temporal adaptada para películas
if "release_date" in df.columns:
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    # 1. Evolución del promedio de calificaciones por año
    df.groupby("release_year")["vote_average"].mean().plot()
    plt.title("Evolución temporal del promedio de calificaciones (vote_average)")
    plt.ylabel("Calificación promedio")
    plt.xlabel("Año de estreno")
    plt.show()

    # 2. Cantidad de películas estrenadas por año
    df["release_year"].value_counts().sort_index().plot(kind="bar")
    plt.title("Cantidad de películas estrenadas por año")
    plt.xlabel("Año de estreno")
    plt.ylabel("Cantidad de películas")
    plt.show()

    # 3. Promedio de cantidad de votos por año
    df.groupby("release_year")["vote_count"].mean().plot()
    plt.title("Evolución temporal del promedio de cantidad de votos")
    plt.ylabel("Cantidad promedio de votos")
    plt.xlabel("Año de estreno")
    plt.show()

# ================================
# 8. CREACIÓN DEL DATASET REDUCIDO
# ================================

# --- 1. Reemplazar ceros por NaN ---
df[["revenue", "vote_average", "vote_count", "budget"]] = df[["revenue", "vote_average", "vote_count", "budget"]].replace(0, np.nan)

# --- 2. Eliminar nulos en columnas clave ---
cols_clave = ["vote_average", "vote_count", "release_date", "genres",
              "production_countries", "spoken_languages", "keywords",
              "directors", "writers", "cast"]

df_limpio = df.dropna(subset=cols_clave).copy()

# --- 3. Quitar columnas innecesarias ---
cols_a_eliminar = ["budget", "revenue", "tagline", "poster_path", "backdrop_path", "homepage"]
df_limpio.drop(columns=cols_a_eliminar, inplace=True, errors="ignore")

print("Tamaño del dataset limpio:", df_limpio.shape)

# --- 4. Quitar outliers en vote_average ---
Q1 = df_limpio["vote_average"].quantile(0.25)
Q3 = df_limpio["vote_average"].quantile(0.75)
IQR = Q3 - Q1
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR
df_limpio = df_limpio[(df_limpio["vote_average"] >= lim_inf) & (df_limpio["vote_average"] <= lim_sup)]

print("Después de limpiar outliers:", df_limpio.shape[0], "registros")

# --- 5. Crear dataset reducido (mínimo 20k registros) ---
if df_limpio.shape[0] < 20000:
    df_reducido = df_limpio.sample(frac=1, random_state=42)  # usar todos si hay menos de 20k
else:
    df_reducido = df_limpio.sample(n=20000, random_state=42)

# --- 6. Guardar el reducido ---
ruta_salida = "dataset_reducido.csv"
df_reducido.to_csv(ruta_salida, index=False)
print("✅ Dataset reducido guardado en:", ruta_salida)
print("Número de registros finales:", df_reducido.shape[0])

# --- 7. Verificación ---
df_verificacion = pd.read_csv(ruta_salida)
print("\nPrimeras filas del dataset reducido:")
print(df_verificacion.head())
