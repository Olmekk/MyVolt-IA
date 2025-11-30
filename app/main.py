from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus

app = FastAPI()

# ==========================================
# ⚡ CONFIGURACIÓN DE CORS (EL PERMISO)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. CONFIGURACIÓN DE BASE DE DATOS (LOCAL)
# ==========================================

DB_HOST = os.getenv("DB_HOST", "host.docker.internal") 
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "alextorres") 
DB_PASS = os.getenv("DB_PASS", "") 

DB_PASS_ENCODED = quote_plus(DB_PASS)

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    engine = create_engine(DATABASE_URL)
    print(f"✅ Configuración de BD Local: Conectando a {DB_HOST}...")
except Exception as e:
    print(f"⚠️ Error configurando BD: {e}")
    engine = None

# ==========================================
# 2. CARGA DE MODELOS (TV LED)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../notebooks/neural_network/RN_DT_tv_led_real_autoencoder_v1.h5')
SCALER_PATH = os.path.join(BASE_DIR, '../notebooks/scaler/DT_tv_led_real_scaler_v1.pkl')
THRESHOLD_PATH = os.path.join(BASE_DIR, '../notebooks/umbral/tv_led_threshold.txt')

model_tv = None
scaler_tv = None
UMBRAL_RECONSTRUCTION = 0.05 

UMBRAL_CONSUMO_NORMAL = 80.0 # 🎯 La regla es correcta aquí

print("🔄 Cargando modelos...")
try:
    model_tv = load_model(MODEL_PATH, compile=False)
    scaler_tv = joblib.load(SCALER_PATH)
    
    try:
        with open(THRESHOLD_PATH, 'r') as f:
            file_umbral = float(f.read().strip())
            if file_umbral >= 0.04:
                UMBRAL_RECONSTRUCTION = file_umbral
    except:
        print(f"⚠️ No se pudo cargar el umbral desde el archivo. Usando el valor por defecto: {UMBRAL_RECONSTRUCTION}")

    print(f"✅ Modelos y Umbral de Reconstrucción ({UMBRAL_RECONSTRUCTION}) cargados.")
except Exception as e:
    print(f"❌ Error cargando modelos (Verifica que existan los archivos): {e}")

# ==========================================
# 3. ENDPOINT
# ==========================================
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
    if model_tv is None:
        raise HTTPException(status_code=500, detail="Modelos no cargados")

    device_clean = data.device_type.strip()
    power_value = data.power_w 
    
    print(f"🔍 Analizando solicitud para: '{device_clean}' con {power_value}W") 

    if device_clean == "TV LED":
        
        # 🎯 REGLA DE NEGOCIO: SI CONSUMO < 80W, ES NORMAL (SIN ANOMALÍA)
        if power_value < UMBRAL_CONSUMO_NORMAL:
            is_anomaly = False
            loss = 0.0
            mensaje_humano = "✅ Todo en orden. Consumo bajo, funcionamiento correcto."
            
        else:
            # Si el consumo es alto (>= 80W), corremos el modelo de IA
            try:
                # Pre-procesamiento
                power_value_float = float(power_value)
                input_scaled = scaler_tv.transform(pd.DataFrame([[power_value_float]], columns=['power']))
                
                # Inferencia
                reconstruction = model_tv.predict(input_scaled, verbose=0)
                
                # Error
                loss = mean_absolute_error(input_scaled, reconstruction)
                
                # Decisión: ¿La reconstrucción falló?
                is_anomaly = bool(loss > UMBRAL_RECONSTRUCTION)
                
                # Mensaje personalizado
                if is_anomaly:
                    mensaje_humano = f"⚠️ ¡Cuidado! Se detectó una anomalía. El consumo de {power_value}W es inusual para tu TV."
                else:
                    mensaje_humano = "✅ Todo en orden. Funcionamiento correcto."

            except Exception as e:
                 raise HTTPException(status_code=500, detail=f"Error interno en el procesamiento del modelo: {str(e)}")


        # Respuesta Final
        return {
            "device": "TV LED",
            "input_watts": power_value,
            "reconstruction_error": float(loss),
            "threshold": UMBRAL_RECONSTRUCTION,
            "is_anomaly": is_anomaly,
            "mensaje": mensaje_humano,
            "status": "ALERTA" if is_anomaly else "OK"
        }
    else:
        return {
            "error": "Modelo no disponible",
            "recibido": device_clean,
            "esperado": "TV LED"
        }