# PROYECTO FINAL - PREDICCIÓN DE PRECIOS DE VIVIENDA
## DESCRIPCIÓN
Este proyecto tiene como objetivo predecir el precio medio de viviendas en California utilizando técnicas de Machine Learning. Se desarrolló un pipeline completo que abarca desde la exploración de datos hasta el despliegue del modelo en una API REST, asegurando consistencia entre entrenamiento e inferencia.

## PIPELINE DEL PROYECTO
1. Exploración de datos (EDA)
   - Análisis de distribuciones y detección de outliers  
   - Evaluación de correlaciones entre variables  
   - Identificación de variables con mayor poder explicativo  
2. Limpieza y Feature Engineering
   - Imputación de valores faltantes en `total_bedrooms`  
   - Transformación logarítmica de `median_income`  
   - Codificación one-hot de `ocean_proximity`  
   - Estandarización del pipeline para train e inferencia  
 3. Modelado
Se evaluaron múltiples modelos:
   - Linear Regression  
   - Decision Tree  
   - Random Forest  
Optimización mediante **GridSearchCV** con validación cruzada.

## MODELO GANADOR
- Modelo: RandomForestRegressor  
- Features: Variables originales (Set A)  
- Criterio de selección: Mejor desempeño en validación cruzada con equilibrio entre sesgo y varianza  

## RESULTADOS
   - RMSE ~50,000
   - MAE ~33,000
   - R² ~0.82
El modelo presenta buen desempeño general y capacidad para capturar relaciones no lineales en los datos.

## INTERPRETACION 
El modelo captura adecuadamente relaciones no lineales entre variables.
El error absoluto es significativo, pero razonable en el contexto del mercado inmobiliario.
El feature engineering adicional no mejoró el desempeño, lo que indica que el modelo ya extrae patrones complejos de las variables originales.
El modelo presenta un buen equilibrio entre sesgo y varianza, sin evidencias claras de sobreajuste.

## API
Se implementó una API REST con FastAPI para servir el modelo.
   # Ejecutar API
uvicorn src.api.main:app --reload
   # Documentación interactiva
http://127.0.0.1:8000/docs
   # Ejemplo de request
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
   # Ejemplo de respuesta
{
  "predicted_price": 455073.19
}

## ESTRUCTURA DEL PROYECTO 
src/
├── api/             # API FastAPI
├── data/            # Scripts de carga y partición de datos
├── features/        # Feature engineering
├── models/          # Entrenamiento y serialización del modelo

## REPRODUCTIBILIDAD
Para ejecutar el proyecto desde cero:
   - 1. Generar dataset
python -m src.data.make_dataset
   - 2. Dividir datos
python -m src.data.split_data
   - 3. Entrenar modelo
python -m src.models.train_model
   - 4. Levantar API
uvicorn src.api.main:app --reload

## CONCLUSION 
El modelo Random Forest demostró ser el más robusto para este problema, capturando relaciones no lineales sin evidencias significativas de sobreajuste.
La implementación de una API permite trasladar el modelo a un entorno de producción, facilitando su integración en aplicaciones reales y la generación de predicciones en tiempo real.

## AUTHOR
Tais RODRIGUEZ