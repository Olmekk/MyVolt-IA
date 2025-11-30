from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus

app = FastAPI()

# ==========================================
# CONFIGURACIÓN DE CORS (EL PERMISO)
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
# DB_HOST = os.getenv("DB_HOST", "host.docker.internal") 
# DB_PORT = os.getenv("DB_PORT", "5432")
# DB_NAME = os.getenv("DB_NAME", "postgres")
# DB_USER = os.getenv("DB_USER", "alextorres") 
# DB_PASS = os.getenv("DB_PASS", "")

DB_USER = "postgres"
DB_PASS = "1234"
DB_HOST = "host.docker.internal"
DB_PORT = "5432"
DB_NAME = "postgres_2"

DB_PASS_ENCODED = quote_plus(DB_PASS)
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    engine = create_engine(DATABASE_URL)
    print(f"Configuración de BD Local: Conectando a {DB_HOST}...")
except Exception as e:
    print(f"Error configurando BD: {e}")
    engine = None

# ==========================================
# 2. CARGA DE MODELOS (TV LED)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- RUTA ---
MODEL_PATH = os.path.join(BASE_DIR, '../notebooks/models/models_v2/RN_DT_tv_led__real_v2.pkl')
# ------------------------------------------

model_if = None
UMBRAL_CONSUMO_NORMAL = 65.0 # Regla

print(f"Buscando modelo en: {os.path.abspath(MODEL_PATH)}")

try:
    # Cargar ML TV LED
    model_if = joblib.load(MODEL_PATH)
    print("¡ML (V2) cargado exitosamente!")
except Exception as e:
    print(f"Error fatal cargando modelo: {e}")

# ==========================================
# 3. ENDPOINT
# ==========================================
class ConsumptionData(BaseModel):
    device_type: str
    power_w: float 
    voltage: float = 120.0
    current_a: float = 0.0

@app.get("/")
def read_root():
    return {"status": "MyVolt IA Service Online"}

@app.post("/predict/anomaly")
def predict_anomaly(data: ConsumptionData):
    if model_if is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado. Revisa logs.")

    device_clean = data.device_type.strip()
    power_value = float(data.power_w)
    
    print(f"Analizando solicitud para: '{device_clean}' con {power_value}W") 

    if device_clean == "TV LED":
        
        # SI CONSUMO < 65W, ES NORMAL
        if power_value < UMBRAL_CONSUMO_NORMAL:
            is_anomaly = False
            mensaje_humano = "Todo en orden. Consumo bajo, funcionamiento correcto."
            
        else:
            # Si el consumo es alto (>= 65W), usamos el modelo Isolation Forest
            try:
                # Pre-procesamiento
                input_df = pd.DataFrame([[power_value]], columns=['power_w'])
                
                # Inferencia
                resultado_numpy = model_if.predict(input_df)[0]
                
                # --- CORRECCIÓN CLAVE ---
                # Convertimos de NumPy a Python nativo para que no falle el JSON
                prediccion = int(resultado_numpy)
                is_anomaly = bool(prediccion == -1)
                # ------------------------
                
                # Mensaje personalizado
                if is_anomaly:
                    mensaje_humano = f"¡Cuidado! Se detectó una anomalía. El consumo de {power_value}W es inusual para tu TV."
                else:
                    mensaje_humano = "Todo en orden. Consumo dentro del patrón normal de la TV."

            except Exception as e:
                 # Imprimimos el error en la terminal de Docker para verlo
                 print(f"Error en predicción IA: {e}")
                 raise HTTPException(status_code=500, detail=f"Error interno en el modelo: {str(e)}")
             
        # Respuesta Final
        return {
            "device": "TV LED",
            "input_watts": power_value,
            "algorithm": "ML V2",
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