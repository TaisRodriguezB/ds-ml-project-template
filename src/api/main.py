"""
API básica usando FastAPI para servir el modelo entrenado.
"""

from __future__ import annotations

import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from src.features.build_features import preprocess_features


app = FastAPI(
    title="API de Predicción de Precios de Vivienda (California)",
    version="1.0"
)


class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


artifact = None
model = None
bedrooms_median = None
feature_columns = None


@app.on_event("startup")
def load_model():
    global artifact, model, bedrooms_median, feature_columns

    artifact = joblib.load("models/best_model.pkl")
    model = artifact["model"]
    bedrooms_median = artifact["bedrooms_median"]
    feature_columns = artifact["feature_columns"]


@app.get("/")
def home():
    return {"mensaje": "Bienvenido a la API del Proyecto Final de Ciencia de Datos"}


@app.post("/predict")
def predict_price(features: HousingFeatures):
    if model is None:
        return {"error": "El modelo no se ha cargado."}

    # 1. convertir entrada a DataFrame
    input_df = pd.DataFrame([features.model_dump()])

    # 2. aplicar el mismo preprocesamiento que en entrenamiento
    processed_df, _ = preprocess_features(input_df, bedrooms_median=bedrooms_median)

    # 3. asegurar mismas columnas y orden que en entrenamiento
    processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)

    # 4. predecir
    prediction = model.predict(processed_df)[0]

    return {"predicted_price": float(prediction)}