from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import random
from sqlalchemy import create_engine
from urllib.parse import quote_plus
app = FastAPI()

# configuracion de cors el permiso
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. configuracion de base de datos local
# db_host = os.getenv("DB_HOST", "host.docker.internal") 
# db_port = os.getenv("DB_PORT", "5432")
# db_name = os.getenv("DB_NAME", "postgres")
# db_user = os.getenv("DB_USER", "alextorres") 
# db_pass = os.getenv("DB_PASS", "")

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

# 2. carga de modelos diccionario dinamico
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ruta
MODELS_DIR = os.path.join(BASE_DIR, '../notebooks/models/models_v2/')

# mapeo: dispositivo a archivo pkl, limite maximo watts y limite maximo en reposo
DEVICE_CONFIG = {
    # Cocina
    "cafetera": {"file": "modelo_cafetera.pkl", "max_w": 1200.0, "standby_max": 3.0},
    "licuadora": {"file": "modelo_licuadora.pkl", "max_w": 1000.0, "standby_max": 1.0},
    "microondas": {"file": "modelo_microondas.pkl", "max_w": 1200.0, "standby_max": 4.0},
    "horno de microondas": {"file": "modelo_microondas.pkl", "max_w": 1200.0, "standby_max": 4.0},
    "horno microondas": {"file": "modelo_microondas.pkl", "max_w": 1200.0, "standby_max": 4.0},
    "refrigerador": {"file": "modelo_refrigerador.pkl", "max_w": 600.0, "standby_max": 15.0},
    "tostador pan": {"file": "modelo_tostador_pan.pkl", "max_w": 1600.0, "standby_max": 2.0},
    "tostadora": {"file": "modelo_tostador_pan.pkl", "max_w": 1600.0, "standby_max": 2.0},

    # Cuidado Personal
    "plancha ropa": {"file": "modelo_plancha_ropa.pkl", "max_w": 2400.0, "standby_max": 1.0},
    "plancha de ropa": {"file": "modelo_plancha_ropa.pkl", "max_w": 2400.0, "standby_max": 1.0},
    "plancha": {"file": "modelo_plancha_ropa.pkl", "max_w": 2400.0, "standby_max": 1.0},
    "secadora pelo": {"file": "modelo_secadora_pelo.pkl", "max_w": 2000.0, "standby_max": 1.0},
    "secadora de pelo": {"file": "modelo_secadora_pelo.pkl", "max_w": 2000.0, "standby_max": 1.0},
    "secadora": {"file": "modelo_secadora_pelo.pkl", "max_w": 2000.0, "standby_max": 1.0},

    # Electrónica
    "computadora": {"file": "modelo_computadora.pkl", "max_w": 800.0, "standby_max": 5.0},
    "computadora escritorio": {"file": "modelo_computadora.pkl", "max_w": 800.0, "standby_max": 5.0},
    "computadora de escritorio": {"file": "modelo_computadora.pkl", "max_w": 800.0, "standby_max": 5.0},
    "laptop": {"file": "modelo_laptop.pkl", "max_w": 90.0, "standby_max": 3.0},
    "laptop (cargando)": {"file": "modelo_laptop.pkl", "max_w": 90.0, "standby_max": 3.0},
    "tv led": {"file": "modelo_tv_led.pkl", "max_w": 60.0, "standby_max": 3.0},
    "tv": {"file": "modelo_tv_led.pkl", "max_w": 60.0, "standby_max": 3.0},
    "consola videojuegos": {"file": "modelo_consola_videojuegos.pkl", "max_w": 250.0, "standby_max": 15.0},
    "consola de videojuegos": {"file": "modelo_consola_videojuegos.pkl", "max_w": 250.0, "standby_max": 15.0},
    "consola": {"file": "modelo_consola_videojuegos.pkl", "max_w": 250.0, "standby_max": 15.0},

    # Climatización
    "calefactor portatil": {"file": "modelo_calefactor_portatil.pkl", "max_w": 1500.0, "standby_max": 4.0},
    "calefactor": {"file": "modelo_calefactor_portatil.pkl", "max_w": 1500.0, "standby_max": 4.0},
    "ventilador de pedestal": {"file": "modelo_ventilador_pedestal.pkl", "max_w": 90.0, "standby_max": 2.0},
    "ventilador pedestal": {"file": "modelo_ventilador_pedestal.pkl", "max_w": 90.0, "standby_max": 2.0},
    "ventilador": {"file": "modelo_ventilador_pedestal.pkl", "max_w": 90.0, "standby_max": 2.0}
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

# 3. endpoint
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
    # .strip() quita espacios al inicio/final, .lower() ignora Mayúsculas
    device_clean = data.device_type.strip().lower()
    power_value = float(data.power_w)
    
    # Buscamos en el diccionario normalizado
    if device_clean not in DEVICE_CONFIG:
        print(f"ERROR: No encontré el dispositivo '{device_clean}'")
        return {
            "error": "Dispositivo desconocido",
            "recibido": device_clean,
            "status": "ERROR"
        }

    # obtener configuracion dinamica
    model_if = LOADED_MODELS[device_clean]
    limite_seguridad = DEVICE_CONFIG[device_clean]["max_w"]
    limite_reposo = DEVICE_CONFIG[device_clean]["standby_max"]

    print(f"Analizando solicitud para: '{device_clean}' con {power_value}W") 

    # si consumo supera el limite es anomalia por regla de seguridad
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

    # proceso de inferencia ia
    try:
        # preprocesamiento
        input_df = pd.DataFrame([[power_value]], columns=['power_w'])
        
        # inferencia
        resultado_numpy = model_if.predict(input_df)[0]
        
        # convertimos de numpy a python nativo para que no falle el json
        prediccion = int(resultado_numpy)
        is_anomaly = bool(prediccion == -1)
        
        # mensaje personalizado
        if is_anomaly:
            mensaje_humano = f"Alerta! Se detecto una anomalia. El consumo de {power_value}W es inusual para: {device_clean}."
        else:
            mensaje_humano = f"Todo en orden. Consumo dentro del patron normal para: {device_clean}."

        # llamada al motor de reglas definido abajo
        obj_recomendacion = ejecutar_reglas_experto(
            device_clean, 
            power_value, 
            is_anomaly, 
            limite_seguridad,
            limite_reposo
        )

    except Exception as e:
            # imprimimos el error en la terminal de docker para verlo
            print(f"Error en prediccion IA ({device_clean}): {e}")
            raise HTTPException(status_code=500, detail=f"Error interno en el modelo: {str(e)}")
        
    # respuesta final
    return {
        "device": device_clean,
        "input_watts": power_value,
        "algorithm": "IsolationForest V2",
        "is_anomaly": is_anomaly,
        "mensaje": mensaje_humano,
        "recomendacion": obj_recomendacion, 
        "status": "ALERTA" if is_anomaly else "OK"
    }

# 4. motor basado en reglas sistema experto
# memoria ram para recordar historial reciente
DEVICE_STATUS_MEMORY = {}

def ejecutar_reglas_experto(device_type, current_watts, is_anomaly_ia, max_limit, standby_max_limit):
    """
    analiza el historial inmediato para dar consejos utiles
    retorna un diccionario con la recomendacion o none
    """
    
    # inicializar memoria si es nuevo
    if device_type not in DEVICE_STATUS_MEMORY:
        DEVICE_STATUS_MEMORY[device_type] = {
            'conteo_anomalias': 0,   
            'ciclos_alto_consumo': 0, 
            'ciclos_standby': 0      
        }
    
    stats = DEVICE_STATUS_MEMORY[device_type]
    recomendacion = None 

    # regla a: mantenimiento e inestabilidad
    if is_anomaly_ia:
        stats['conteo_anomalias'] += 1
    else:
        if stats['conteo_anomalias'] > 0:
            stats['conteo_anomalias'] -= 1 
    
    # si acumula 5 errores recientes
    if stats['conteo_anomalias'] >= 5:
        mensajes_mantenimiento = [
            f"El dispositivo {device_type} ha presentado comportamiento inestable frecuente recientemente.",
            f"Hemos notado que tu {device_type} ha tenido varias fluctuaciones extrañas. Seria bueno revisarlo.",
            f"Atencion: el {device_type} registro multiples anomalias seguidas. Sugerimos un chequeo preventivo.",
            f"El patron de energia de tu {device_type} es muy inestable hoy. Podria necesitar mantenimiento.",
            f"Detectamos picos anormales repetidos en tu {device_type}. Te sugerimos verificar su estado fisico.",
            f"Las lecturas de tu {device_type} no son normales. Quiza sea momento de darle una revision profunda.",
            f"Tu {device_type} esta teniendo variaciones constantes de energia. Esto podria indicar una falla interna.",
            f"El consumo de tu {device_type} esta subiendo y bajando de forma atipica. Mantenlo vigilado.",
            f"Registramos actividad sospechosa y constante en tu {device_type}. Un tecnico deberia revisarlo.",
            f"La inestabilidad de tu {device_type} continua. Recomendamos desconectarlo y checarlo pronto."
        ]
        
        recomendacion = {
            "titulo": "Revision Recomendada",
            "mensaje": random.choice(mensajes_mantenimiento),
            "tipo": "MANTENIMIENTO", 
            "prioridad": "ALTA"
        }
        stats['conteo_anomalias'] = 0 
        return recomendacion

    # regla b: sobrecarga y fatiga
    # si usa mas del 85 por ciento de su potencia maxima
    umbral_fatiga = max_limit * 0.85
    
    if current_watts > umbral_fatiga:
        stats['ciclos_alto_consumo'] += 1
    else:
        stats['ciclos_alto_consumo'] = 0 
    
    # 150 ciclos son aprox 5 minutos continuos al maximo si envias cada 2s
    if stats['ciclos_alto_consumo'] > 150:
        mensajes_sobrecarga = [
            f"El dispositivo lleva mucho tiempo operando al limite ({int(current_watts)}W). Podria fatigarse.",
            f"Tu {device_type} lleva un buen rato trabajando al maximo. Cuidado con el sobrecalentamiento.",
            f"Precaucion: uso intenso prolongado en el {device_type}. Considera darle un descanso pronto.",
            f"El {device_type} esta operando al limite de su capacidad. Vigila su temperatura para evitar daños.",
            f"Notamos que tu {device_type} no ha bajado su consumo de {int(current_watts)}W. Pausarlo alargara su vida util.",
            f"Alerta de fatiga: el {device_type} sigue consumiendo mucha energia. Es buen momento para apagarlo un rato.",
            f"Trabajar a {int(current_watts)}W por tanto tiempo puede desgastar tu {device_type}. Te sugerimos un respiro.",
            f"El esfuerzo continuo de tu {device_type} es alto. Apagarlo unos minutos ayudara a enfriar sus componentes.",
            f"Tu {device_type} lleva demasiado tiempo exigiendo el maximo de energia. Evita un accidente termico.",
            f"Consumo tope detectado por tiempo prolongado en tu {device_type}. Recomendamos hacer una pausa."
        ]
        
        recomendacion = {
            "titulo": "Posible Sobrecalentamiento",
            "mensaje": random.choice(mensajes_sobrecarga),
            "tipo": "USO", 
            "prioridad": "MEDIA"
        }
        stats['ciclos_alto_consumo'] = 0
        return recomendacion

    # regla c: consumo vampiro para ahorro dinamico
    # verificamos si esta consumiendo algo mayor a cero pero menor o igual a su limite maximo de reposo
    if 0.0 < current_watts <= standby_max_limit:
        stats['ciclos_standby'] += 1
    else:
        stats['ciclos_standby'] = 0 
    
    # 900 ciclos son aprox 30 minutos detectando consumo hormiga
    if stats['ciclos_standby'] > 900:
        mensajes_vampiro = [
            f"Tu {device_type} parece estar en espera gastando energia inutilmente. Desconectalo si no lo usas.",
            f"Detectamos que tu {device_type} esta en modo reposo. Desconectalo de la pared para ahorrar energia.",
            f"El {device_type} esta consumiendo energia fantasma. Si ya terminaste de usarlo, apagalo por completo.",
            f"Pequeño aviso: el {device_type} lleva rato consumiendo un nivel bajo. Desenchufarlo ayudara a tu recibo.",
            f"Tu {device_type} sigue conectado pero inactivo. Evita el consumo vampiro quitandolo del enchufe.",
            f"Hay un gasto silencioso de energia en tu {device_type}. Desconectarlo de la clavija cuidara tu bolsillo.",
            f"El {device_type} esta en modo de espera hace bastante. Recuerda que conectado sigue consumiendo luz.",
            f"Apagar del boton no siempre es suficiente. Tu {device_type} sigue gastando, te conviene desconectarlo.",
            f"Detectamos consumo hormiga en tu {device_type}. Quitalo de la corriente para un ahorro total.",
            f"Tu {device_type} no esta en uso, pero sigue jalando energia. Desenchufalo y suma ahorro a tu mes."
        ]
        
        recomendacion = {
            "titulo": "Consumo Vampiro Detectado",
            "mensaje": random.choice(mensajes_vampiro),
            "tipo": "AHORRO", 
            "prioridad": "BAJA"
        }
        stats['ciclos_standby'] = 0 
        return recomendacion

    return None

    return None