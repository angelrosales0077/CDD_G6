import os
import shutil
import kagglehub

def descargar_dataset(destino_carpeta=r"E:\CDD\TP CDD Git\CDD_G6",
                      nombre_csv="TMDB IMDB Movies Dataset.csv"):
    """
    Descarga el dataset de Kaggle usando kagglehub y lo guarda en la carpeta destino.
    Devuelve la ruta completa del CSV.
    """
    # Crear carpeta destino si no existe
    os.makedirs(destino_carpeta, exist_ok=True)
    destino = os.path.join(destino_carpeta, nombre_csv)

    # Descargar dataset
    path_cache = kagglehub.dataset_download("ggtejas/tmdb-imdb-merged-movies-dataset")
    print("Dataset descargado en:", path_cache)

    # Copiar CSV desde cache a destino
    for archivo in os.listdir(path_cache):
        if archivo.endswith(".csv"):
            origen = os.path.join(path_cache, archivo)
            shutil.copy(origen, destino)
            print(f"Archivo copiado a: {destino}")
            break

    # Borrar carpeta cache
    shutil.rmtree(path_cache, ignore_errors=True)
    print("Carpeta cache eliminada ✅")

    return destino
