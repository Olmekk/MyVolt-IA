from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import os

app = FastAPI()

# 1. Configuración de Rutas de Modelos
# Usamos rutas relativas para encontrar la carpeta 'models'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../notebooks/neural_network/RN_DT_tv_led_real_autoencoder_v1.h5')
SCALER_PATH = os.path.join(BASE_DIR, '../notebooks/scaler/DT_tv_led_real_scaler_v1.pkl')

# 2. Cargar el Modelo y el Escalador al iniciar la API
# Esto se hace una sola vez para que sea rápido
print("Cargando modelos de IA...")
try:
    model_tv = load_model(MODEL_PATH, compile=False)    
    scaler_tv = joblib.load(SCALER_PATH)
    print("¡Modelos cargados exitosamente!")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    model_tv = None
    scaler_tv = None

# 3. Cargar el Umbral Dinámico (Desde el archivo que guardaste)
# Definimos la ruta relativa al archivo de texto
# Asumiendo que está en: MyVolt-IA/notebooks/umbral/tv_led_threshold.txt
THRESHOLD_PATH = os.path.join(BASE_DIR, '../notebooks/umbral/tv_led_threshold.txt')

print(f"Cargando umbral desde: {THRESHOLD_PATH}")

try:
    with open(THRESHOLD_PATH, 'r') as f:
        # Leemos el texto y lo convertimos a flotante
        UMBRAL_TV = float(f.read().strip())
    print(f"¡Umbral cargado exitosamente: {UMBRAL_TV}!")
except Exception as e:
    print(f"Error cargando umbral (usando valor por defecto): {e}")
    UMBRAL_TV = 0.05 # Valor de seguridad por si falla el archivo

# 4. Definir la estructura de los datos que recibiremos
# Esto valida que el backend nos mande los datos correctos
class ConsumptionData(BaseModel):
    device_type: str
    power_w: float
    voltage: float
    current_a: float

@app.get("/")
def read_root():
    return {"status": "MyVolt IA Service Online"}

@app.post("/predict/anomaly")
def predict_anomaly(data: ConsumptionData):
    """
    Recibe datos de consumo y determina si es una anomalía.
    """
    if model_tv is None:
        raise HTTPException(status_code=500, detail="Modelos no cargados")

    # Lógica de Enrutamiento (Router)
    # Aquí decidimos qué modelo usar según el aparato
    if data.device_type == "TV LED":
        
        # A. Pre-procesamiento
        # Convertimos el dato de entrada a un DataFrame
        input_df = pd.DataFrame([[data.power_w]], columns=['power_w'])
        
        # Escalamos el dato (de Watts a 0-1) usando el scaler guardado
        input_scaled = scaler_tv.transform(input_df)
        
        # B. Inferencia (El modelo intenta "dibujar" el dato)
        reconstruction = model_tv.predict(input_scaled, verbose=0)
        
        # C. Cálculo de Error (Loss)
        # Calculamos qué tan mal lo dibujó (MAE)
        loss = mean_absolute_error(input_scaled, reconstruction)
        
        # D. Decisión
        # Si el error es mayor al umbral, es anomalía
        is_anomaly = bool(loss > UMBRAL_TV)
        
        return {
            "device": "TV LED",
            "input_watts": data.power_w,
            "reconstruction_error": float(loss),
            "threshold": UMBRAL_TV,
            "is_anomaly": is_anomaly,
            "status": "ALERTA" if is_anomaly else "OK"
        }
    
    else:
        return {"error": "Modelo no disponible para este dispositivo"}