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
    allow_origins=["*"],  # Permite que CUALQUIER App (Flutter, Web, etc.) se conecte
    allow_credentials=True,
    allow_methods=["*"],  # Permite TODOS los métodos (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Permite TODOS los headers
)

# ==========================================
# 1. CONFIGURACIÓN DE BASE DE DATOS (LOCAL)
# ==========================================

# 'host.docker.internal' es la dirección mágica para que Docker vea tu Windows
DB_HOST = os.getenv("DB_HOST", "host.docker.internal") 
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres") 
DB_PASS = os.getenv("DB_PASS", "1234") # Contraseña por defecto (se sobrescribe con -e en Docker)

# Codificamos la contraseña
DB_PASS_ENCODED = quote_plus(DB_PASS)

# Cadena de conexión LOCAL (Sin sslmode)
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
UMBRAL_TV = 0.05 

print("🔄 Cargando modelos...")
try:
    model_tv = load_model(MODEL_PATH, compile=False)
    scaler_tv = joblib.load(SCALER_PATH)
    with open(THRESHOLD_PATH, 'r') as f:
        UMBRAL_TV = float(f.read().strip())
    print(f"✅ Modelos y Umbral ({UMBRAL_TV}) cargados.")
except Exception as e:
    print(f"❌ Error cargando modelos (Verifica que existan los archivos): {e}")

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
    if model_tv is None:
        raise HTTPException(status_code=500, detail="Modelos no cargados")

    # --- MEJORA 1: Limpieza de texto ---
    # Quitamos espacios en blanco al inicio o final para evitar errores tontos
    device_clean = data.device_type.strip()
    
    print(f"🔍 Analizando solicitud para: '{device_clean}' con {data.power_w}W")

    # Router de Dispositivos
    if device_clean == "TV LED":
        try:
            # Pre-procesamiento
            input_scaled = scaler_tv.transform(pd.DataFrame([[data.power_w]], columns=['power_w']))
            
            # Inferencia
            reconstruction = model_tv.predict(input_scaled, verbose=0)
            
            # Error
            loss = mean_absolute_error(input_scaled, reconstruction)
            
            # Decisión
            is_anomaly = bool(loss > UMBRAL_TV)
            
            # Mensaje personalizado
            mensaje_humano = ""
            if is_anomaly:
                mensaje_humano = f"⚠️ ¡Cuidado! Se detectó una anomalía. El consumo de {data.power_w}W es inusual para tu TV."
            else:
                mensaje_humano = "✅ Todo en orden. No se han detectado anomalías en tu TV."

            return {
                "device": "TV LED",
                "input_watts": data.power_w,
                "reconstruction_error": float(loss),
                "threshold": UMBRAL_TV,
                "is_anomaly": is_anomaly,
                "mensaje": mensaje_humano,
                "status": "ALERTA" if is_anomaly else "OK"
            }
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
    else:
        # --- MEJORA 2: Error detallado ---
        return {
            "error": "Modelo no disponible",
            "recibido": device_clean,
            "esperado": "TV LED"
        }