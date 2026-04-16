"""
Módulo para limpieza y enriquecimiento mínimo de datos
alineado con el modelo final ganador.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "median_house_value"
CATEGORICAL_COLUMN = "ocean_proximity"


def clean_data(df: pd.DataFrame, bedrooms_median: float | None = None) -> tuple[pd.DataFrame, float]:
    """
    Limpia el DataFrame imputando valores faltantes en total_bedrooms.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    bedrooms_median : float | None
        Mediana de total_bedrooms calculada sobre train.
        Si no se proporciona, se calcula sobre el propio df.

    Returns
    -------
    tuple[pd.DataFrame, float]
        DataFrame limpio y mediana utilizada.
    """
    df = df.copy()

    if bedrooms_median is None:
        bedrooms_median = df["total_bedrooms"].median()

    df["total_bedrooms"] = df["total_bedrooms"].fillna(bedrooms_median)

    return df, bedrooms_median


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea únicamente las transformaciones finales coherentes
    con el modelo ganador.

    En este caso:
    - log transform de median_income

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame limpio.

    Returns
    -------
    pd.DataFrame
        DataFrame enriquecido.
    """
    df = df.copy()

    df["median_income_log"] = np.log1p(df["median_income"])

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica one-hot encoding a ocean_proximity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Returns
    -------
    pd.DataFrame
        DataFrame con variables categóricas codificadas.
    """
    df = df.copy()

    if CATEGORICAL_COLUMN in df.columns:
        df = pd.get_dummies(df, columns=[CATEGORICAL_COLUMN], drop_first=True)

    return df


def align_features(train_df: pd.DataFrame, other_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Alinea columnas entre train y otro DataFrame (test o inferencia).

    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame de referencia.
    other_df : pd.DataFrame
        DataFrame a alinear.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Ambos DataFrames con mismas columnas y mismo orden.
    """
    train_df = train_df.copy()
    other_df = other_df.copy()

    train_df, other_df = train_df.align(other_df, join="left", axis=1, fill_value=0)

    return train_df, other_df


def preprocess_features(
    df: pd.DataFrame,
    bedrooms_median: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Orquesta el preprocesamiento completo de features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame crudo.
    bedrooms_median : float | None
        Mediana de total_bedrooms calculada en train.

    Returns
    -------
    tuple[pd.DataFrame, float]
        DataFrame procesado y mediana usada.
    """
    df_clean, bedrooms_median = clean_data(df, bedrooms_median=bedrooms_median)
    df_feat = create_features(df_clean)
    df_encoded = encode_categorical(df_feat)

    return df_encoded, bedrooms_median


def split_xy(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa variables predictoras y variable objetivo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame procesado.
    target_column : str
        Nombre de la variable objetivo.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X e y.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


if __name__ == "__main__":
    print("Módulo de feature engineering listo.")
