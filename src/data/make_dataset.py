import os
import tarfile
import urllib.request
from pathlib import Path


def fetch_housing_data(housing_url: str, housing_path: str):
    # Crear carpeta si no existe
    Path(housing_path).mkdir(parents=True, exist_ok=True)

    tgz_path = os.path.join(housing_path, "housing.tgz")

    # Descargar archivo
    urllib.request.urlretrieve(housing_url, tgz_path)

    # Extraer archivo
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

    print("Datos descargados y extraídos correctamente")


if __name__ == "__main__":
    URL = "https://github.com/ageron/data/raw/main/housing.tgz"
    PATH = "data/raw/"
    fetch_housing_data(URL, PATH)