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
    print(f"Configuracion de BD Local: Conectando a {DB_HOST}...")
except Exception as e:
    print(f"Error configurando BD: {e}")
    engine = None

# ==========================================
# 2. CARGA DE MODELOS (DICCIONARIO DINAMICO)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- RUTA ---
MODELS_DIR = os.path.join(BASE_DIR, '../notebooks/models/models_v2/')
# ------------------------------------------

# Mapeo: Dispositivo -> (Archivo PKL, Limite Maximo Watts basado en tu lista)
DEVICE_CONFIG = {
    "Cafetera": {"file": "modelo_cafetera.pkl", "max_w": 1200.0},
    "Calefactor Portatil": {"file": "modelo_calefactor_portatil.pkl", "max_w": 1500.0},
    "Computadora": {"file": "modelo_computadora.pkl", "max_w": 800.0},
    "Consola Videojuegos": {"file": "modelo_consola_videojuegos.pkl", "max_w": 250.0},
    "Laptop": {"file": "modelo_laptop.pkl", "max_w": 90.0},
    "Licuadora": {"file": "modelo_licuadora.pkl", "max_w": 1000.0},
    "Microondas": {"file": "modelo_microondas.pkl", "max_w": 1200.0},
    "Plancha Ropa": {"file": "modelo_plancha_ropa.pkl", "max_w": 2400.0},
    "Refrigerador": {"file": "modelo_refrigerador.pkl", "max_w": 600.0},
    "Secadora Pelo": {"file": "modelo_secadora_pelo.pkl", "max_w": 2000.0},
    "Tostador Pan": {"file": "modelo_tostador_pan.pkl", "max_w": 1600.0},
    "TV LED": {"file": "modelo_tv_led.pkl", "max_w": 60.0},
    "Ventilador Pedestal": {"file": "modelo_ventilador_pedestal.pkl", "max_w": 90.0}
}

LOADED_MODELS = {}

print(f"Buscando modelos en: {os.path.abspath(MODELS_DIR)}")

for device_name, config in DEVICE_CONFIG.items():
    model_path = os.path.join(MODELS_DIR, config["file"])
    try:
        if os.path.exists(model_path):
            LOADED_MODELS[device_name] = joblib.load(model_path)
            print(f"Modelo cargado: {device_name}")
        else:
            print(f"ARCHIVO NO ENCONTRADO: {config['file']}")
    except Exception as e:
        print(f"Error fatal cargando {device_name}: {e}")

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
    return {
        "status": "MyVolt IA Service Online",
        "loaded_models": list(LOADED_MODELS.keys())
    }

@app.post("/predict/anomaly")
def predict_anomaly(data: ConsumptionData):
    
    device_clean = data.device_type.strip()
    power_value = float(data.power_w)
    
    if device_clean not in LOADED_MODELS:
        return {
            "error": "Modelo no disponible o dispositivo desconocido",
            "recibido": device_clean,
            "status": "ERROR"
        }

    # OBTENER CONFIGURACION
    model_if = LOADED_MODELS[device_clean]
    limite_seguridad = DEVICE_CONFIG[device_clean]["max_w"]

    print(f"Analizando solicitud para: '{device_clean}' con {power_value}W") 

    # SI CONSUMO SUPERA EL LIMITE, ES ANOMALIA (REGLA SEGURIDAD)
    if power_value > limite_seguridad:
        return {
            "device": device_clean,
            "input_watts": power_value,
            "is_anomaly": True,
            "mensaje": f"Peligro! El consumo de {power_value}W supera el limite fisico de {limite_seguridad}W.",
            "recomendacion": {
                "titulo": "Peligro Electrico",
                "mensaje": "Desconecta el dispositivo inmediatamente. Riesgo de corto o falla critica.",
                "tipo": "SEGURIDAD",
                "prioridad": "CRITICA"
            },
            "status": "PELIGRO"
        }

    # PROCESO DE INFERENCIA IA
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
            mensaje_humano = f"Alerta! Se detecto una anomalia. El consumo de {power_value}W es inusual para: {device_clean}."
        else:
            mensaje_humano = f"Todo en orden. Consumo dentro del patron normal para: {device_clean}."

        # === LLAMADA AL MOTOR DE REGLAS (DEFINIDO ABAJO) ===
        obj_recomendacion = ejecutar_reglas_experto(
            device_clean, 
            power_value, 
            is_anomaly, 
            limite_seguridad
        )
        # ===================================================

    except Exception as e:
            # Imprimimos el error en la terminal de Docker para verlo
            print(f"Error en prediccion IA ({device_clean}): {e}")
            raise HTTPException(status_code=500, detail=f"Error interno en el modelo: {str(e)}")
        
    # Respuesta Final
    return {
        "device": device_clean,
        "input_watts": power_value,
        "algorithm": "IsolationForest V2",
        "is_anomaly": is_anomaly,
        "mensaje": mensaje_humano,
        "recomendacion": obj_recomendacion, # CAMPO NUEVO
        "status": "ALERTA" if is_anomaly else "OK"
    }

# ==========================================
# 4. MOTOR BASADO EN REGLAS (SISTEMA EXPERTO)
# ==========================================
# Memoria RAM para recordar historial reciente
DEVICE_STATUS_MEMORY = {}

def ejecutar_reglas_experto(device_type, current_watts, is_anomaly_ia, max_limit):
    """
    Analiza el historial inmediato para dar consejos utiles.
    Retorna: Un diccionario con la recomendacion o None.
    """
    
    # Inicializar memoria si es nuevo
    if device_type not in DEVICE_STATUS_MEMORY:
        DEVICE_STATUS_MEMORY[device_type] = {
            'conteo_anomalias': 0,   
            'ciclos_alto_consumo': 0, 
            'ciclos_standby': 0      
        }
    
    stats = DEVICE_STATUS_MEMORY[device_type]
    recomendacion = None 

    # --- REGLA A: MANTENIMIENTO (Inestabilidad) ---
    if is_anomaly_ia:
        stats['conteo_anomalias'] += 1
    else:
        if stats['conteo_anomalias'] > 0:
            stats['conteo_anomalias'] -= 1 
    
    # Si acumula 5 errores recientes
    if stats['conteo_anomalias'] >= 5:
        recomendacion = {
            "titulo": "Revision Recomendada",
            "mensaje": f"El dispositivo {device_type} ha presentado comportamiento inestable frecuente recientemente.",
            "tipo": "MANTENIMIENTO", 
            "prioridad": "ALTA"
        }
        stats['conteo_anomalias'] = 0 
        return recomendacion

    # --- REGLA B: SOBRECARGA (Fatiga) ---
    # Si usa mas del 85% de su potencia maxima
    umbral_fatiga = max_limit * 0.85
    
    if current_watts > umbral_fatiga:
        stats['ciclos_alto_consumo'] += 1
    else:
        stats['ciclos_alto_consumo'] = 0 
    
    # 150 ciclos = aprox 5 minutos continuos al maximo (si envias cada 2s)
    if stats['ciclos_alto_consumo'] > 150:
        recomendacion = {
            "titulo": "Posible Sobrecalentamiento",
            "mensaje": f"El dispositivo lleva mucho tiempo operando al limite ({int(current_watts)}W). Podria fatigarse.",
            "tipo": "USO", 
            "prioridad": "MEDIA"
        }
        stats['ciclos_alto_consumo'] = 0
        return recomendacion

    # --- REGLA C: CONSUMO VAMPIRO (Ahorro) ---
    # Si consume entre 0.5W y 5W
    if 0.5 < current_watts < 5.0:
        stats['ciclos_standby'] += 1
    else:
        stats['ciclos_standby'] = 0 
    
    # 900 ciclos = aprox 30 minutos detectando consumo hormiga
    if stats['ciclos_standby'] > 900:
        recomendacion = {
            "titulo": "Consumo Vampiro Detectado",
            "mensaje": f"Tu {device_type} parece estar en espera gastando energia inutilmente. Desconectalo si no lo usas.",
            "tipo": "AHORRO", 
            "prioridad": "BAJA"
        }
        stats['ciclos_standby'] = 0 
        return recomendacion

    return None