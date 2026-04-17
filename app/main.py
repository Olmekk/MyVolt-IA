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
    if stats['conteo_anomalias'] >= 10:
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
            f"La inestabilidad de tu {device_type} continua. Recomendamos desconectarlo y checarlo pronto.",
            f"Las variaciones de energia en tu {device_type} no son normales. Revisa que el cable y el enchufe esten en buen estado.",
            f"Tu {device_type} esta mostrando un patron de consumo erratico. Te sugerimos no usarlo sin supervision.",
            f"Hemos detectado saltos electricos inusuales en tu {device_type}. Evita un posible cortocircuito revisandolo a tiempo.",
            f"El comportamiento de tu {device_type} es impredecible hoy. Considera consultar a un tecnico especializado.",
            f"Cuidado: tu {device_type} registra picos de corriente que podrian dañar su placa interna.",
            f"Parece que a tu {device_type} le esta costando estabilizar su consumo. Un mantenimiento preventivo es ideal ahora.",
            f"La IA sigue detectando ruido electrico en tu {device_type}. Podria ser un sintoma de desgaste natural.",
            f"Tu {device_type} esta jalando energia de forma muy dispareja. Asegurate de que no este haciendo falso contacto.",
            f"Alerta de inestabilidad prolongada. Desconecta tu {device_type} y verifica si huele a quemado o hace ruidos extraños.",
            f"Las metricas de tu {device_type} estan fuera de su huella normal. Te recomendamos darle servicio tecnico.",
            f"El historial de tu {device_type} muestra demasiada variabilidad. Seria prudente revisar sus conexiones internas.",
            f"Tu {device_type} no logra mantener un flujo de energia constante. Esto acelera el deterioro de su fuente de poder.",
            f"Hemos captado un patron de parpadeo electrico en tu {device_type}. Mantenlo bajo vigilancia estricta hoy.",
            f"Los saltos de corriente en tu {device_type} persisten. Quiza el enchufe de la pared este fallando, revisalo.",
            f"Tu {device_type} esta operando fuera de sus parametros normales de forma repetida. Requiere atencion tecnica."
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
            f"Consumo tope detectado por tiempo prolongado en tu {device_type}. Recomendamos hacer una pausa.",
            f"Tu {device_type} lleva operando a {int(current_watts)}W sostenidos por mucho tiempo. Protegelo del estres termico.",
            f"El nivel de exigencia de tu {device_type} no ha bajado. Darle un respiro ahora extendera su vida util.",
            f"Detectamos carga pesada continua en tu {device_type}. Pausar su uso prevendra fallas por calor.",
            f"Mantener tu {device_type} a {int(current_watts)}W constantes puede derretir componentes internos. Apagalo un momento.",
            f"El motor o sistema de tu {device_type} esta trabajando sin descanso. Un ciclo de enfriamiento es altamente recomendado.",
            f"Llevas un periodo largo exigiendo el limite a tu {device_type}. Cuida tu inversion y dejalo reposar.",
            f"Atencion: uso severo prolongado. Si tu {device_type} se siente muy caliente al tacto, desconectalo de inmediato.",
            f"El consumo de {int(current_watts)}W se ha mantenido demasiado tiempo. Evita que tu {device_type} se queme por fatiga.",
            f"Tu {device_type} esta dando el 100% desde hace rato. Reducir la carga de trabajo prevendra un daño permanente.",
            f"Aviso de seguridad termica: tu {device_type} necesita un receso despues de este ciclo de alto consumo.",
            f"Alerta preventiva: tu {device_type} no ha tenido descanso. Reducir su uso ahora prevendra fallas criticas.",
            f"El estres electrico en tu {device_type} es muy alto. Apagalo un rato para que sus circuitos internos respiren.",
            f"Tu {device_type} esta operando al limite durante mucho tiempo. El exceso de calor es el enemigo numero uno.",
            f"Advertencia de sobrecarga continua. Desconectar tu {device_type} unos minutos es la mejor practica ahora mismo.",
            f"Notamos un consumo maximo sostenido en tu {device_type}. Usarlo asi por horas reduce drasticamente su vida util."
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
    if stats['ciclos_standby'] > 250:
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
            f"Tu {device_type} no esta en uso, pero sigue jalando energia. Desenchufalo y suma ahorro a tu mes.",
            f"Aunque no lo uses, tu {device_type} sigue sumando centavos a tu recibo. Desenchufalo para un ahorro real.",
            f"Tu {device_type} esta en modo pasivo chupando electricidad. Usa un multicontacto con interruptor para apagarlo facil.",
            f"Cortar la corriente de tu {device_type} ahora mismo evitara que siga generando un gasto invisible.",
            f"El indicador led o la fuente de tu {device_type} sigue consumiendo. Ayuda al planeta y a tu cartera desconectandolo.",
            f"Cuidado con los consumos ocultos. Tu {device_type} lleva horas en reposo pero aun conectado a la red.",
            f"¿Terminaste de usar tu {device_type}? Sacarlo del enchufe es el ultimo paso para ser verdaderamente eficiente.",
            f"Tu {device_type} registra fuga pasiva de energia. Estos pequeños consumos suman mucho al final del bimestre.",
            f"Para detener al 100% el medidor de luz, necesitas desconectar tu {device_type}, no solo apagarlo.",
            f"Tu {device_type} esta en standby extendido. Desconectalo y convierte ese desperdicio en ahorro.",
            f"La IA nota que tu {device_type} sigue activo a nivel minimo. Quitale la energia si no planeas usarlo pronto.",
            f"El enchufe sigue enviando energia a tu {device_type} en modo reposo. Desconectalo y empieza a ahorrar.",
            f"Tu {device_type} esta apagado pero su transformador interno sigue consumiendo luz. Quitalo de la corriente.",
            f"Evita sorpresas en tu recibo electrico. Tu {device_type} esta inactivo pero consumiendo energia constante.",
            f"Pequeñas fugas de energia detectadas en tu {device_type}. Desenchufalo por completo para maximizar la eficiencia.",
            f"El modo espera de tu {device_type} esta sumando watts innecesarios. Un pequeño esfuerzo de desconectarlo ayuda mucho."
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