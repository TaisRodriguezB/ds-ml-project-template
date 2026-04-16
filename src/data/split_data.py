import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


def split_and_save_data(raw_data_path: str, interim_data_path: str):
    # Crear carpeta de salida si no existe
    Path(interim_data_path).mkdir(parents=True, exist_ok=True)

    # Cargar datos
    df = pd.read_csv(raw_data_path)

    # Crear variable categórica para estratificar por ingreso
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, float("inf")],
        labels=[1, 2, 3, 4, 5]
    )

    # Split estratificado
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["income_cat"]):
        train_set = df.loc[train_index].copy()
        test_set = df.loc[test_index].copy()

    # Quitar variable auxiliar
    for dataset in (train_set, test_set):
        dataset.drop("income_cat", axis=1, inplace=True)

    # Guardar
    train_set.to_csv(f"{interim_data_path}/train.csv", index=False)
    test_set.to_csv(f"{interim_data_path}/test.csv", index=False)

    print("Split estratificado completado y guardado.")
    print(f"Train shape: {train_set.shape}")
    print(f"Test shape: {test_set.shape}")


if __name__ == "__main__":
    RAW_PATH = "data/raw/housing/housing.csv"
    INTERIM_PATH = "data/interim/"
    split_and_save_data(RAW_PATH, INTERIM_PATH)