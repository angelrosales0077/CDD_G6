import kagglehub
import os

def descargar_dataset():
    """
    Descarga el dataset de Kaggle usando kagglehub y devuelve la ruta al archivo CSV.
    kagglehub gestiona la caché, por lo que no es necesario copiar ni borrar nada.
    """
    print("Descargando dataset desde Kaggle Hub...")
    
    # KAGGLEHUB_DOWNLOAD_PATH devuelve la carpeta donde se descargó el dataset
    path_cache = kagglehub.dataset_download("ggtejas/tmdb-imdb-merged-movies-dataset")
    
    # Buscamos el archivo CSV dentro de la carpeta descargada
    for archivo in os.listdir(path_cache):
        if archivo.endswith(".csv"):
            ruta_csv = os.path.join(path_cache, archivo)
            print(f"✅ Dataset disponible en: {ruta_csv}")
            return ruta_csv
            
    # Si no se encuentra el CSV, lanzamos un error
    raise FileNotFoundError("No se encontró ningún archivo .csv en el dataset descargado.")