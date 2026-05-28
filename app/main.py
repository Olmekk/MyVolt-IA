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
# db_port = os.getenv("DB_PORT", "5433")
# db_name = os.getenv("DB_NAME", "postgres")
# db_user = os.getenv("DB_USER", "") 
# db_pass = os.getenv("DB_PASS", "")
# configuracion de base de datos servidor postgres
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

# carga de modelos diccionario dinamico
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ruta
MODELS_DIR = os.path.join(BASE_DIR, '../notebooks/models/models_v2/')

# mapeo dispositivo a archivo pkl limite maximo watts y limite maximo en reposo
DEVICE_CONFIG = {
    # cocina
    "cafetera": {"file": "modelo_cafetera.pkl", "max_w": 1200.0, "standby_max": 3.0},
    "licuadora": {"file": "modelo_licuadora.pkl", "max_w": 1000.0, "standby_max": 1.0},
    "microondas": {"file": "modelo_microondas.pkl", "max_w": 1200.0, "standby_max": 4.0},
    "horno de microondas": {"file": "modelo_microondas.pkl", "max_w": 1200.0, "standby_max": 4.0},
    "horno microondas": {"file": "modelo_microondas.pkl", "max_w": 1200.0, "standby_max": 4.0},
    "refrigerador": {"file": "modelo_refrigerador.pkl", "max_w": 600.0, "standby_max": 15.0},
    "tostador pan": {"file": "modelo_tostador_pan.pkl", "max_w": 1600.0, "standby_max": 2.0},
    "tostadora": {"file": "modelo_tostador_pan.pkl", "max_w": 1600.0, "standby_max": 2.0},

    # cuidado personal
    "plancha ropa": {"file": "modelo_plancha_ropa.pkl", "max_w": 2400.0, "standby_max": 1.0},
    "plancha de ropa": {"file": "modelo_plancha_ropa.pkl", "max_w": 2400.0, "standby_max": 1.0},
    "plancha": {"file": "modelo_plancha_ropa.pkl", "max_w": 2400.0, "standby_max": 1.0},
    "secadora pelo": {"file": "modelo_secadora_pelo.pkl", "max_w": 2000.0, "standby_max": 1.0},
    "secadora de pelo": {"file": "modelo_secadora_pelo.pkl", "max_w": 2000.0, "standby_max": 1.0},
    "secadora": {"file": "modelo_secadora_pelo.pkl", "max_w": 2000.0, "standby_max": 1.0},

    # electronica
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

    # climatizacion
    "calefactor portatil": {"file": "modelo_calefactor_portatil.pkl", "max_w": 1500.0, "standby_max": 4.0},
    "calefactor": {"file": "modelo_calefactor_portatil.pkl", "max_w": 1500.0, "standby_max": 4.0},
    "ventilador de pedestal": {"file": "modelo_ventilador_pedestal.pkl", "max_w": 160.0, "standby_max": 2.0},
    "ventilador pedestal": {"file": "modelo_ventilador_pedestal.pkl", "max_w": 160.0, "standby_max": 2.0},
    "ventilador": {"file": "modelo_ventilador_pedestal.pkl", "max_w": 160.0, "standby_max": 2.0}
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

# endpoint de inferencia
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
    # quita espacios al inicio y final e ignora mayusculas
    device_clean = data.device_type.strip().lower()
    power_value = float(data.power_w)
    
    # buscamos en el diccionario
    if device_clean not in DEVICE_CONFIG:
        print(f"ERROR: No se encontró el dispositivo '{device_clean}'")
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
        
        # opciones de texto principal para advertir sobre el wattaje
        mensajes_peligro = [
            f"Peligro inminente: el consumo de {power_value}W es demasiado alto para tu {device_clean} (limite: {limite_seguridad}W).",
            f"Alerta critica: tu {device_clean} esta jalando {power_value}W, superando su capacidad maxima de {limite_seguridad}W.",
            f"Sobrecarga detectada. El registro de {power_value}W rompe el limite de seguridad de {limite_seguridad}W para tu {device_clean}.",
            f"Emergencia electrica: {power_value}W excede por completo el tope seguro de {limite_seguridad}W en tu {device_clean}.",
            f"Cuidado, tu {device_clean} alcanzo un pico de {power_value}W. Su limite fisico es de solo {limite_seguridad}W.",
            f"Pico de energia critico. Tu {device_clean} reporta {power_value}W, lo cual supera los {limite_seguridad}W permitidos.",
            f"Lectura peligrosa de {power_value}W en tu {device_clean}. Esto esta muy por encima de sus {limite_seguridad}W maximos.",
            f"Alerta maxima: el enchufe detecto {power_value}W, rebasando el umbral seguro de {limite_seguridad}W para tu {device_clean}.",
            f"Nivel de consumo inseguro. Tu {device_clean} marco {power_value}W frente a un tope de {limite_seguridad}W.",
            f"Falla de seguridad termica: {power_value}W superan el limite estructural de {limite_seguridad}W en tu {device_clean}.",
            f"Sobrecarga grave en el enchufe. Tu {device_clean} solicita {power_value}W, mas alla de sus {limite_seguridad}W seguros.",
            f"Atencion: tu {device_clean} esta consumiendo {power_value}W, violando la restriccion fisica de {limite_seguridad}W.",
            f"Advertencia critica de consumo. {power_value}W es un valor excesivo para el limite de {limite_seguridad}W de tu {device_clean}.",
            f"Riesgo estructural: el consumo de {power_value}W destruira tu {device_clean} si supera asi sus {limite_seguridad}W.",
            f"Alerta de corto. Tu {device_clean} registro {power_value}W, rompiendo por completo la barrera de {limite_seguridad}W."
        ]

        # opciones de texto secundario para indicar la accion que debe tomar el usuario
        recomendaciones_peligro = [
            "Desconecta el dispositivo inmediatamente. Riesgo inminente de cortocircuito o incendio.",
            "Corta la corriente de inmediato. El equipo podria fundirse o causar daños graves.",
            "Por tu seguridad, desenchufa el aparato en este instante. Hay peligro de fuego.",
            "Apaga y desconecta el equipo ya. No lo vuelvas a encender sin que un tecnico lo revise.",
            "Riesgo de daño permanente. Retira la clavija de la pared lo mas pronto posible.",
            "Peligro de accidente electrico. Aleja el equipo de materiales inflamables y desconectalo.",
            "Quita la energia del enchufe de inmediato. Los componentes internos podrian estar derritiendose.",
            "Accion requerida urgente: desenchufa el aparato para evitar que tu instalacion electrica sufra daños.",
            "Situacion de riesgo. Baja la pastilla de la luz o desconecta el equipo de la pared ya mismo.",
            "Para evitar un incendio, desconecta el aparato inmediatamente. No lo sigas utilizando.",
            "Sobrecarga extrema. Retira la conexion electrica al instante para proteger tu hogar.",
            "Desenchufa este electrodomestico de urgencia. Mantenerlo conectado es un riesgo critico.",
            "Evita una catastrofe electrica. Corta el suministro a este enchufe de inmediato.",
            "Apagado de emergencia sugerido. Desconecta fisicamente el equipo antes de que genere humo.",
            "Falla critica inminente. Quita el cable de la corriente para salvar el aparato y la instalacion."
        ]

        return {
            "device": device_clean,
            "input_watts": power_value,
            "is_anomaly": True,
            "mensaje": random.choice(mensajes_peligro),
            "recomendacion": {
                "titulo": "Peligro Electrico",
                "mensaje": random.choice(recomendaciones_peligro),
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
        
        # listas de mensajes para consumo anomalo
        mensajes_ia_anomalia = [
            f"Alerta! Se detecto una anomalia. El consumo de {power_value}W es inusual para: {device_clean}.",
            f"Hemos detectado un pico anomalo de {power_value}W en tu {device_clean}.",
            f"El comportamiento de tu {device_clean} salio de su patron normal marcando {power_value}W.",
            f"Cuidado, la Inteligencia Artificial detecto {power_value}W como una lectura atipica en tu {device_clean}.",
            f"Anomalia registrada. Tu {device_clean} esta consumiendo {power_value}W de forma inesperada.",
            f"Aviso importante, el registro de {power_value}W no cuadra con el historial de tu {device_clean}.",
            f"El modelo aislo el dato de {power_value}W como una posible falla en tu {device_clean}.",
            f"Detectamos actividad sospechosa, tu {device_clean} esta jalando {power_value}W.",
            f"La lectura de {power_value}W en tu {device_clean} es estadisticamente anormal.",
            f"Precaucion, el consumo de {power_value}W en tu {device_clean} rompio su tendencia habitual.",
            f"Tu {device_clean} reporto un gasto atipico de {power_value}W, mantente atento.",
            f"El sistema de IA marco el consumo de {power_value}W como irregular para tu {device_clean}.",
            f"Hay una desviacion en el consumo de tu {device_clean}, alcanzo los {power_value}W sin razon aparente.",
            f"Alerta de telemetria, el dato de {power_value}W en tu {device_clean} es considerado una anomalia.",
            f"La red electrica de tu {device_clean} muestra un comportamiento raro al llegar a {power_value}W."
        ]

        # listas de mensajes para consumo normal
        mensajes_ia_normal = [
            f"Todo en orden. Consumo dentro del patron normal para: {device_clean}.",
            f"Funcionamiento optimo, tu {device_clean} mantiene un patron seguro de {power_value}W.",
            f"El consumo de {power_value}W esta dentro del historial esperado para tu {device_clean}.",
            f"Lectura correcta, tu {device_clean} opera sin problemas a {power_value}W.",
            f"Sin novedades, el registro de {power_value}W de tu {device_clean} es completamente normal.",
            f"La IA confirma que el gasto de {power_value}W es habitual para tu {device_clean}.",
            f"Parametros estables, tu {device_clean} sigue trabajando a {power_value}W sin alteraciones.",
            f"El dato de {power_value}W encaja perfecto en la huella electrica de tu {device_clean}.",
            f"Monitoreo limpio, el consumo de {power_value}W de tu {device_clean} esta en el rango seguro.",
            f"Tu {device_clean} se comporta maravillosamente con un consumo de {power_value}W.",
            f"Todo tranquilo, el requerimiento de {power_value}W es tipico de tu {device_clean}.",
            f"El algoritmo evaluo {power_value}W en tu {device_clean} y lo marco como un dato sano.",
            f"Operacion normal detectada, los {power_value}W de tu {device_clean} no representan riesgo.",
            f"Lectura de rutina exitosa, el consumo de {power_value}W para tu {device_clean} es el adecuado.",
            f"Estado verde, tu {device_clean} fluye con una energia estable de {power_value}W."
        ]

        # seleccion dinamica del mensaje humano segun la prediccion
        if is_anomaly:
            mensaje_humano = random.choice(mensajes_ia_anomalia)
        else:
            mensaje_humano = random.choice(mensajes_ia_normal)

        # llamada al motor de reglas
        obj_recomendacion = ejecutar_reglas_experto(
            device_clean, 
            power_value, 
            is_anomaly, 
            limite_seguridad,
            limite_reposo
        )

    except Exception as e:
            # imprimimos el error en la terminal de docker
            print(f"Error en el análisis con IA ({device_clean}): {e}")
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

# motor basado en reglas sistema experto
# memoria ram para recordar historial reciente
DEVICE_STATUS_MEMORY = {}

# declaracion mensajes para el mantenimiento preventivo, recomendaciones de uso y ahorro segun el dispositivo
msg_tv = {
    "mantenimiento": [
        "Hola, notamos variaciones de energía en tu TV. Revisa que el cable esté bien conectado.",
        "¡Cuidado! Tu pantalla registra inestabilidad eléctrica. Un regulador podría protegerla.",
        "Detectamos parpadeos eléctricos en tu TV. Asegúrate de que el enchufe no tenga falso contacto.",
        "Las lecturas de tu TV son inusuales. Revisa la clavija para evitar daños en la fuente de poder.",
        "Tu TV está recibiendo energía de forma irregular. Te sugerimos revisar la conexión.",
        "Alerta preventiva: los picos de voltaje podrían dañar los LEDs de tu pantalla. Échale un ojo.",
        "Registramos ruido eléctrico en tu TV. Conectarla a un supresor de picos es una gran idea.",
        "El flujo de corriente en tu tele no es constante. Verifica que no haya cables en mal estado.",
        "Tu pantalla muestra un comportamiento eléctrico atípico. Un chequeo rápido evitará sorpresas.",
        "Hay pequeñas fallas de energía en tu TV. Mantente alerta para evitar un cortocircuito."
    ],
    "uso": [
        "Parece que tu TV lleva muchas horas encendida. ¡Dale un respiro para cuidar su panel!",
        "¡Maratón detectado! Apagar tu tele un momento ayudará a enfriar sus componentes internos.",
        "Tu TV ha estado trabajando al máximo por un buen rato. Pausarla prolongará su vida útil.",
        "Cuidemos esa pantalla. Lleva mucho tiempo activa, un pequeño descanso le vendría genial.",
        "El consumo de tu TV lleva horas al límite. Te recomendamos apagarla unos 15 minutos.",
        "Para evitar retención de imagen o desgaste de pixeles, te sugerimos apagar tu tele un rato.",
        "Tu TV está generando bastante calor interno por el uso continuo. ¡Es hora de un descanso!",
        "Llevas una sesión muy larga en la pantalla. Apagarla ahora previene el estrés térmico.",
        "Notamos uso intensivo en tu tele. Darle un respiro optimiza su rendimiento a largo plazo.",
        "Tu pantalla ha trabajado duro hoy. Apagarla un momento es la mejor forma de cuidarla."
    ],
    "ahorro": [
        "Tu TV está apagada pero el foquito sigue consumiendo. ¡Desconéctala y ahorra en tu recibo!",
        "Tip de ahorro: desconectar tu tele de la pared evita el consumo vampiro por completo.",
        "El modo de espera de tu pantalla gasta energía silenciosamente. Desenchúfala si no la usas.",
        "Aunque no la veas, tu TV sigue sumando a tu recibo eléctrico. Quítala de la corriente.",
        "¡Corta el consumo invisible! Desconecta tu TV y nota la diferencia a fin de mes.",
        "Tu tele sigue conectada a la red eléctrica. Desenchufarla es un hábito súper ecológico.",
        "El modo de inicio rápido de tu pantalla gasta luz todo el día. Apágalo o desconéctala.",
        "Para un ahorro real, apagar con el control no basta. ¡Desenchufa tu TV de la pared!",
        "Detectamos consumo fantasma en tu tele. Quítale la energía para maximizar tu ahorro.",
        "Tu TV está en modo pasivo. Desconectarla ahora mismo es dinero que se queda en tu bolsa."
    ]
}

msg_ventilador = {
    "mantenimiento": [
        "Tu ventilador registra saltos de energía. Podría necesitar limpieza en el eje del motor.",
        "Notamos inestabilidad en tu ventilador. Verifica que las aspas giren libremente sin atorarse.",
        "El motor de tu ventilador hace picos raros. Una lubricada rápida podría evitar que se queme.",
        "Alerta de hardware: tu ventilador jala corriente de forma atípica. Revisa si tiene mucho polvo.",
        "El capacitor de tu ventilador podría estar fallando. Sugerimos un chequeo de mantenimiento.",
        "Las variaciones eléctricas indican que al motor le cuesta girar. Límpialo para evitar daños.",
        "Tu ventilador muestra resistencia eléctrica. Asegúrate de que nada esté bloqueando las aspas.",
        "Detectamos ruido eléctrico inusual. Revisar el enchufe y el motor extenderá su vida útil.",
        "El consumo de tu ventilador es errático hoy. Presta atención a si hace ruidos mecánicos extraños.",
        "Cuidado, hay fluctuaciones en el motor de tu ventilador. Un servicio preventivo le caería bien."
    ],
    "uso": [
        "Tu ventilador lleva mucho tiempo a máxima velocidad. Bájale un poco para cuidar el embobinado.",
        "El motor del ventilador debe estar bastante caliente. ¡Apágalo unos minutos para que respire!",
        "Uso continuo prolongado. Pausar tu ventilador ahora evitará que sus plásticos se derritan.",
        "Le estás exigiendo mucho a tu ventilador. Darle un descanso es clave para que dure más veranos.",
        "Tu ventilador ha estado al máximo por horas. Reduce la velocidad para evitar sobrecalentamiento.",
        "El estrés térmico en tu ventilador es alto. Un ciclo de enfriamiento es justo lo que necesita.",
        "Para proteger el motor de tu ventilador, te sugerimos apagarlo por un periodo corto de tiempo.",
        "Trabajar sin pausas desgasta tu equipo. Apaga tu ventilador un rato y cuida tu inversión.",
        "Notamos que tu ventilador no ha descansado. Haz una pausa para evitar que se queme internamente.",
        "El uso extremo de hoy podría fatigar tu ventilador. ¡Déjalo reposar un momento!"
    ],
    "ahorro": [
        "Tu ventilador está apagado pero enchufado. Desconéctalo si ya refrescó la habitación.",
        "Tip de ahorro: el ventilador sigue energizado. Quítalo de la pared y cuida tu economía.",
        "Aunque las aspas no giren, tu ventilador consume una pizca de luz. ¡Desenchúfalo!",
        "Corta el consumo vampiro de raíz. Si no hace calor, desconecta tu ventilador por completo.",
        "Mantener el ventilador conectado sin usarlo suma a tu recibo. ¡Desconéctalo y ahorra!",
        "Tu ventilador registra consumo pasivo. Quitarlo del enchufe es un hábito muy eficiente.",
        "Si ya terminaste de usar tu ventilador, dar el paso extra de desconectarlo marca la diferencia.",
        "Ahorro inteligente: desenchufa tu ventilador en las noches frías para no gastar de más.",
        "Detectamos consumo fantasma en tu ventilador de pedestal. ¡Libéralo de la corriente eléctrica!",
        "Para que tu medidor no avance sin sentido, desconecta tu ventilador cuando no lo ocupes."
    ]
}

msg_laptop = {
    "mantenimiento": [
        "La energía hacia tu laptop es inestable. Verifica que el cable del cargador no esté trozado.",
        "Notamos fluctuaciones al cargar tu laptop. Esto podría degradar la salud de tu batería.",
        "Cuidado con el cargador de tu laptop, los picos de voltaje sugieren un falso contacto.",
        "El adaptador de corriente registra variaciones. Revisa que el cuadrito no se sobrecaliente.",
        "Inestabilidad eléctrica detectada. Asegúrate de que el puerto de carga de tu laptop esté limpio.",
        "Hay ruido en la línea de tu laptop. Conectarla a un buen regulador cuidará su tarjeta madre.",
        "Tu equipo recibe energía de forma muy errática. Revisa la instalación de ese enchufe.",
        "Alerta preventiva: los saltos de corriente podrían dañar los circuitos internos de tu laptop.",
        "El consumo de tu cargador no es normal. Considera probar con otro para descartar fallas.",
        "Protege tu información. Las fallas de energía repetidas pueden afectar tu disco duro."
    ],
    "uso": [
        "Tu laptop lleva horas consumiendo su máxima potencia. Cuida que no se sobrecaliente.",
        "Sesión intensa detectada. Asegúrate de que las salidas de aire de tu laptop no estén bloqueadas.",
        "El procesador de tu laptop está trabajando al límite. ¡Dale un respiro al equipo!",
        "Tu cargador está entregando energía a tope desde hace rato. Pausar tu trabajo ayudará a enfriarlo.",
        "Exigencia máxima sostenida. Considera cerrar programas pesados para aliviar tu laptop.",
        "Tu laptop está bajo mucho estrés térmico. Usar una base enfriadora sería una excelente idea.",
        "Llevas mucho tiempo exigiendo el máximo a tu batería y procesador. ¡Haz una pausa!",
        "El consumo al límite podría provocar estrangulamiento térmico. Deja descansar tu laptop.",
        "Cuidemos tu equipo: el uso continuo tan alto acorta la vida de los componentes. Apágala un rato.",
        "Tu laptop necesita enfriarse. Reducir la carga de trabajo ahora prevendrá daños internos."
    ],
    "ahorro": [
        "Tu laptop ya está cargada al 100% pero el cargador sigue consumiendo. ¡Desconéctalo!",
        "Tip de ahorro: dejar el cargador enchufado sin la laptop gasta energía. Quítalo de la pared.",
        "Tu laptop está en suspensión pero sigue gastando luz. Apágala por completo si ya no la usarás.",
        "El consumo vampiro de tu cargador suma centavos a tu recibo. Desenchúfalo para ahorrar.",
        "Cortemos el gasto invisible. Si ya terminaste de trabajar, desconecta tu equipo de la red.",
        "Tu equipo registra consumo pasivo continuo. Apagar y desenchufar es tu mejor estrategia.",
        "Aunque tu laptop esté apagada, el transformador sigue activo. ¡Desconéctalo y ahorra!",
        "Ahorro inteligente: desenchufa tu cargador por las noches para maximizar tu eficiencia.",
        "Detectamos consumo fantasma en tu estación de trabajo. Libera el enchufe de tu laptop.",
        "El foquito de tu cargador te indica que sigue gastando luz. Quítalo para cuidar tu bolsillo."
    ]
}

msg_tostador = {
    "mantenimiento": [
        "Inestabilidad en el tostador. Las migajas acumuladas podrían causar un micro corto, ¡límpialo!",
        "Las resistencias de tu tostador muestran variaciones. Revisa que no haya pan atascado.",
        "Cuidado, hay picos de voltaje inusuales al tostar. Revisa el cable de corriente de tu equipo.",
        "El tostador registra un consumo errático. Asegúrate de que el mecanismo baje correctamente.",
        "Notamos fluctuaciones térmicas en tu tostador. Una limpieza profunda prevendrá accidentes.",
        "Tu tostador está jalando corriente a saltos. Podría haber un problema en su placa interna.",
        "Alerta de mantenimiento: las variaciones continuas pueden dañar las bobinas de calor.",
        "Comportamiento eléctrico extraño en tu cocina. Revisa que el enchufe del tostador no esté flojo.",
        "El elemento de calor de tu tostador muestra desgaste. Mantente alerta cuando lo uses.",
        "Protege tu cocina. La inestabilidad de tu tostador sugiere que requiere una revisión técnica."
    ],
    "uso": [
        "Has usado el tostador por varios ciclos seguidos. ¡Déjalo enfriar para cuidar sus plásticos!",
        "Uso intensivo detectado. La carcasa del tostador debe estar muy caliente, ten precaución.",
        "Las resistencias de tu tostador llevan mucho tiempo activas. Apágalo para evitar sobrecargas.",
        "Le estás exigiendo mucho a tu tostador hoy. Una pausa evitará que el metal interno se deforme.",
        "Tu tostador ha estado trabajando al tope. Déjalo reposar unos minutos antes de volver a usarlo.",
        "El calor acumulado es alto. Pausar el uso de tu tostador ahora reduce riesgos de seguridad.",
        "Sesión prolongada de tostado. Cuida tu equipo dejando que sus componentes se ventilen.",
        "El estrés térmico en tu tostador es elevado. Dale un respiro para que te dure muchos años.",
        "Notamos un uso continuo inusual. Recuerda que es un electrodoméstico de uso corto.",
        "Tu tostador necesita enfriarse. Suspende su uso temporalmente para proteger su resistencia."
    ],
    "ahorro": [
        "¡El desayuno terminó! Tu tostador sigue enchufado, desconéctalo por seguridad y ahorro.",
        "Tip clave: los aparatos que generan calor siempre deben desconectarse tras usarse. ¡Desenchúfalo!",
        "Tu tostador está en modo pasivo sumando a tu recibo. Quítalo de la pared para ahorrar.",
        "Evita consumos vampiro y posibles sustos dejando tu tostador desconectado todo el día.",
        "Si ya no vas a tostar más pan, retirar la clavija es el mejor hábito de ahorro en casa.",
        "El circuito de tu tostador sigue cerrado. Desenchúfalo para lograr un consumo cero absoluto.",
        "Ahorro inteligente: desenchufa tus equipos de cocina cuando no los necesites.",
        "Tu tostador registra una fuga ligera de energía. Desconectarlo cuidará tu bolsillo.",
        "Haz que tu recibo baje desconectando los electrodomésticos pequeños como tu tostador.",
        "Un tostador conectado sin uso es dinero desperdiciado. ¡Libera ese enchufe ahora!"
    ]
}

msg_microondas = {
    "mantenimiento": [
        "El magnetrón de tu microondas registra picos inusuales. Mantente alerta por ruidos extraños.",
        "Inestabilidad eléctrica en el microondas. Verifica que el plato interior gire sin obstrucciones.",
        "Hay variaciones de voltaje en tu microondas. Podría haber una falla en el panel táctil.",
        "Los saltos de corriente detectados sugieren que el inversor de tu microondas está sufriendo.",
        "Notamos consumo errático al calentar. Revisa que la puerta esté sellando correctamente.",
        "Alerta preventiva: el transformador de tu microondas muestra desgaste eléctrico.",
        "Tu microondas jala energía de forma irregular. Considera conectarlo a un tomacorriente propio.",
        "Comportamiento atípico en la cocina. Limpiar el interior de tu microondas evita micro cortos.",
        "El sistema de calor de tu microondas está inestable. Requerirá revisión técnica si esto continúa.",
        "Fluctuaciones graves detectadas. Por seguridad, no uses platos con bordes metálicos."
    ],
    "uso": [
        "Tu microondas lleva mucho tiempo continuo al máximo. El magnetrón necesita enfriarse urgente.",
        "Calentar por periodos tan largos seguidos puede sobrecalentar el motor interno de tu microondas.",
        "Uso prolongado detectado. El ventilador trasero debe estar al límite, dale un respiro.",
        "Pausa el uso de tu microondas. Mantenerlo activo tanto tiempo podría derretir sus fusibles.",
        "El nivel de exigencia para tu microondas es muy alto hoy. Déjalo descansar unos 15 minutos.",
        "Tu electrodoméstico está bajo gran estrés térmico. Apágalo un rato para cuidar tu inversión.",
        "Sesión de cocina intensa. Cuidar los ciclos de enfriamiento de tu microondas prolonga su vida.",
        "Has mantenido la potencia máxima demasiado tiempo. Reducir el uso evitará fallas críticas.",
        "Notamos uso excesivo. Tu microondas es para calentamiento rápido, no para periodos tan largos.",
        "El interior de tu microondas debe estar sumamente caliente. Permite que se disipe el calor."
    ],
    "ahorro": [
        "El reloj de tu microondas gasta energía las 24 horas. ¡Si no lo miras, desconéctalo y ahorra!",
        "Tip de ahorro: el modo de espera del microondas consume luz constante. Quítalo del enchufe.",
        "Tu microondas está en pausa pero sumando a tu recibo. Desenchufarlo corta el gasto de raíz.",
        "Evita el consumo vampiro. Conecta tu microondas a un multicontacto con apagador.",
        "Si no vas a calentar nada pronto, desenchufar el microondas es un hábito súper ahorrador.",
        "Tu equipo registra consumo fantasma. Esa pantallita encendida te cuesta dinero cada mes.",
        "Ahorra en tu cocina: desconecta el microondas cuando te vayas de casa o por las noches.",
        "El sensor de la puerta de tu microondas sigue consumiendo. ¡Desconéctalo para ahorrar!",
        "Corta el flujo pasivo de energía. Un microondas desconectado es la clave de la eficiencia.",
        "Tu microondas no está en uso pero tu medidor sigue girando. Desenchúfalo ahora mismo."
    ]
}

msg_calefactor = {
    "mantenimiento": [
        "Peligro: la resistencia del calefactor muestra inestabilidad. Podría haber un cortocircuito.",
        "Alerta de seguridad: el cable de tu calefactor presenta picos, podría estarse calentando de más.",
        "Variaciones graves en el calefactor. Desconéctalo y verifica que no huela a plástico quemado.",
        "El termostato de tu calefactor es errático, no logra mantener el consumo constante.",
        "Notamos saltos de corriente en el calefactor. Asegúrate de que sus rendijas no estén tapadas.",
        "Comportamiento anómalo en el sistema de calor. Usarlo así representa un riesgo de incendio.",
        "Inestabilidad eléctrica pura. Conecta tu calefactor directo a la pared, sin extensiones.",
        "Las lecturas del calefactor son peligrosas. Revisa que no haya polvo acumulado en la parrilla.",
        "Tu aparato de calefacción está fallando. Un técnico debería revisarlo antes del próximo uso.",
        "Fluctuaciones térmicas detectadas. La resistencia de tu equipo muestra un desgaste crítico."
    ],
    "uso": [
        "Peligro de sobrecalentamiento. Tu calefactor lleva demasiado tiempo al máximo, ¡apágalo ya!",
        "Tu habitación ya debe estar cálida. Apaga el calefactor para evitar sobrecargar la instalación.",
        "El uso extremo de hoy podría derretir los componentes internos de tu calefactor. Dale pausa.",
        "Llevas horas exigiendo el límite a tu calefactor. El estrés térmico es altísimo.",
        "Tu equipo está al rojo vivo desde hace rato. Un descanso prevendrá accidentes en casa.",
        "Mantener el calefactor prendido tanto tiempo no es seguro. Reduce su uso de inmediato.",
        "Cuidemos tu seguridad. Apagar el calefactor por unos 30 minutos enfriará sus plásticos.",
        "Uso crítico detectado. Los equipos de calor no deben operar de forma ininterrumpida.",
        "El consumo sostenido a tope daña la vida útil de tu calefactor. ¡Déjalo reposar un poco!",
        "Tu equipo necesita enfriarse urgentemente. Suspende su funcionamiento por precaución."
    ],
    "ahorro": [
        "El calefactor está apagado pero sigue conectado. Por pura seguridad y ahorro, ¡desenchúfalo!",
        "Tip de ahorro: los equipos de clima gastan mucho en reposo. Quita tu calefactor de la pared.",
        "El sensor antivuelco de tu calefactor sigue consumiendo energía. Desconéctalo por completo.",
        "Si ya no hace frío, mantener el calefactor conectado es tirar dinero. ¡Libera ese enchufe!",
        "Evita consumos vampiro altísimos. Desenchufar el calefactor es el mejor consejo hoy.",
        "Tu equipo térmico sigue registrando fuga pasiva. Desconectarlo bajará tu recibo eléctrico.",
        "Para un cero absoluto en consumo, apagar del botón no basta. Quita la clavija de la red.",
        "Ahorro inteligente: guarda tu calefactor desconectado hasta la próxima noche fría.",
        "Detectamos energía fluyendo hacia el calefactor inactivo. Corta de raíz ese gasto hormiga.",
        "Tu medidor sigue registrando actividad por culpa del calefactor. Desenchúfalo ahora."
    ]
}

msg_licuadora = {
    "mantenimiento": [
        "El motor de tu licuadora sufre picos de voltaje. Asegúrate de que las aspas giren sin esfuerzo.",
        "Notamos inestabilidad al licuar. Revisa que la base de la licuadora no tenga líquido derramado.",
        "Los saltos en la corriente sugieren que los carbones del motor de tu licuadora están gastados.",
        "Cuidado con la licuadora. La variación indica que el motor podría estar haciendo corto interno.",
        "Detectamos resistencia atípica en el embobinado de tu licuadora. Limpiar los baleros ayudaría.",
        "El consumo de la licuadora es errático hoy. Verifica que el cople de plástico no esté roto.",
        "Las fluctuaciones son altas. Asegúrate de no llenar el vaso con alimentos demasiado duros.",
        "Comportamiento eléctrico extraño. Revisa que el cable de tu licuadora esté en perfecto estado.",
        "El motor presenta signos de desgaste severo por la inestabilidad registrada. Ten precaución.",
        "Ruido eléctrico en la cocina. El switch de velocidades de tu licuadora podría tener falso contacto."
    ],
    "uso": [
        "Licuar por periodos tan largos puede quemar el motor. ¡Hazlo en pausas cortas!",
        "El motor de tu licuadora se está calentando por el uso continuo. Déjalo descansar un momento.",
        "Uso intensivo detectado. Forzar la licuadora tanto tiempo podría derretir los engranes.",
        "Estás excediendo el ciclo de trabajo normal de la licuadora. Apágala para que baje su temperatura.",
        "La exigencia para tu licuadora es máxima hoy. Una pausa extenderá la vida del aparato.",
        "Tu equipo está sufriendo estrés térmico. Licuar sin descanso daña la bobina del motor.",
        "Le estás exigiendo mucho a la licuadora. Si huele a plástico caliente, apágala de inmediato.",
        "Cuidemos tu electrodoméstico. Reducir la carga de trabajo ahora evitará fallas mecánicas.",
        "Sesión prolongada. Recuerda que la licuadora está diseñada para trabajos muy rápidos.",
        "El motor necesita enfriarse. Suspende el licuado unos minutos para proteger tu inversión."
    ],
    "ahorro": [
        "La licuadora ya no está en uso pero sigue conectada. ¡Desenchúfala para evitar el consumo vampiro!",
        "Por seguridad y para cuidar tu bolsillo, desconecta la licuadora en cuanto termines de usarla.",
        "Tu licuadora está inactiva pero suma centavos a tu recibo eléctrico. Quítala de la pared.",
        "Tip de ahorro: los equipos con motor jamás deben quedarse conectados en modo reposo.",
        "Evita el gasto silencioso en la cocina. Desenchufar la licuadora es un hábito excelente.",
        "Registramos una pequeña fuga de energía hacia tu licuadora. Corta ese consumo pasivo.",
        "Tu licuadora sigue energizada. Desconectarla previene accidentes y baja tu gasto de luz.",
        "Si ya terminaste de cocinar, liberar los enchufes de la cocina te hará ahorrar mes a mes.",
        "Detectamos consumo fantasma en tu licuadora. Haz que el medidor se detenga desenchufándola.",
        "Un electrodoméstico desconectado es la clave de la eficiencia. Quita tu licuadora ahora."
    ]
}

msg_refrigerador = {
    "mantenimiento": [
        "El compresor de tu refri cicla de forma irregular. Podría faltarle gas refrigerante, revísalo.",
        "Inestabilidad eléctrica en el refri. Verifica que los serpentines traseros no estén tapados de polvo.",
        "Notamos saltos de corriente atípicos. El termostato de tu refrigerador podría estar descalibrado.",
        "El motor de tu refri arranca con variaciones muy fuertes. Revisa el voltaje de tu colonia.",
        "El comportamiento de tu refri es errático. Revisa que no haya escarcha bloqueando los ductos.",
        "Alerta preventiva: las fluctuaciones constantes acortan la vida del compresor de tu refri.",
        "Las lecturas indican un esfuerzo atípico del motor. Sugerimos conectarlo a un regulador dedicado.",
        "Tu refrigerador jala energía a tropezones. Podría haber un problema en la tarjeta de control.",
        "Inestabilidad térmica y eléctrica detectada. El empaque de la puerta podría estar dañado.",
        "Cuidado con tu refri, el ruido eléctrico en el compresor indica desgaste mecánico interno."
    ],
    "uso": [
        "El compresor de tu refri lleva horas sin detenerse. Verifica que la puerta esté bien cerrada.",
        "Tu refri está trabajando horas extra. Asegúrate de no haber metido comida muy caliente.",
        "Sobrecarga en el enfriamiento. Revisa que las ventilas internas del refri no estén bloqueadas.",
        "El motor está al límite continuo. Reducir abrir y cerrar la puerta le dará un merecido descanso.",
        "Tu refrigerador no ha hecho pausas. El ambiente debe estar muy caluroso, vigila su temperatura.",
        "Uso intensivo sostenido. El compresor está bajo estrés extremo intentando enfriar los alimentos.",
        "La demanda térmica de tu refri es altísima hoy. Evita meter más cosas a temperatura ambiente.",
        "Cuidemos el compresor. Está trabajando al tope de su capacidad desde hace un buen rato.",
        "Notamos que tu refrigerador no descansa. Asegúrate de que tenga buena ventilación trasera.",
        "El ciclo de enfriamiento se extendió demasiado. Mantenerlo así gasta mucha energía extra."
    ],
    "ahorro": [
        "Tu refri nunca se desconecta, pero si gasta mucho en mínimo, ajusta su temperatura para ahorrar.",
        "Tip de ahorro: el modo deshielo de tu refri gasta de más. Sube un grado el termostato interno.",
        "Un refri eficiente hace pausas donde consume casi cero. El tuyo está jalando mucha energía pasiva.",
        "Para optimizar tu refri, asegúrate de que no pegue el sol directo sobre él, reducirá tu recibo.",
        "El consumo de reposo de tu refri es alto. Limpiar la goma de las puertas mejorará el aislamiento.",
        "Reduce el gasto pasivo acomodando bien los alimentos para que el aire circule sin tanto esfuerzo.",
        "Tu refri está en modo espera pero su consumo es elevadito. Verifica la temperatura ambiente.",
        "Ahorra optimizando: mantener tu refri a 4°C es ideal para el consumo. No lo pongas más frío.",
        "Detectamos que la pausa del motor gasta más de lo ideal. Una limpieza externa podría ayudar.",
        "Haz más eficiente tu refrigerador despegándolo un par de centímetros de la pared trasera."
    ]
}

msg_pc = {
    "mantenimiento": [
        "La fuente de tu PC presenta caídas de voltaje. Cuidado, esto podría dañar tu tarjeta madre.",
        "Fluctuaciones graves en tu PC. Es indispensable conectarla a un No-Break o regulador confiable.",
        "Picos extraños de energía detectados. Revisa que los cables de tu tarjeta de video estén fijos.",
        "La inestabilidad eléctrica frecuente acorta drásticamente la vida de tus discos duros. Ten precaución.",
        "Tu computadora está recibiendo energía muy sucia. Asegúrate de que el enchufe esté bien aterrizado.",
        "Notamos ruido en la corriente hacia tu PC. Revisa que la fuente de poder no esté llena de polvo.",
        "Alerta de hardware: los saltos de energía indican que tu fuente podría estar cerca de su límite.",
        "Comportamiento errático en tu equipo de escritorio. Un mantenimiento preventivo es urgente.",
        "Tu PC registra parpadeos eléctricos. Guarda tu trabajo seguido para no perder información vital.",
        "El desgaste de los capacitores de tu fuente se acelera con estas variaciones. ¡Échale un ojo!"
    ],
    "uso": [
        "La carga de tu procesador y gráfica están al tope. Cuidado con las altas temperaturas internas.",
        "Sesión de alta exigencia detectada. Monitorea el calor de tu PC para evitar el Thermal Throttling.",
        "El consumo al límite de tu computadora lleva horas. Asegúrate de que los ventiladores giren bien.",
        "Tu PC está al máximo de su capacidad sostenida. Una pausa ayudará a enfriar el sistema entero.",
        "Maratón de uso intenso. Si juegas o renderizas, dale un descanso al hardware pronto.",
        "El estrés térmico en tu PC es elevado. Verifica que no haya bloqueos en el flujo de aire frontal.",
        "Has mantenido la computadora al 100% mucho tiempo. Cuidar los ciclos de enfriamiento es vital.",
        "Notamos exigencia extrema continua. Apagar el equipo por media hora cuidará tu inversión.",
        "Tu fuente de poder está entregando su máxima potencia de forma prolongada. ¡Cuidado!",
        "El uso severo de hoy fatiga los componentes. Dale un respiro a tu estación de trabajo."
    ],
    "ahorro": [
        "Tu PC está apagada pero la motherboard y el RGB siguen chupando luz. ¡Apaga el switch de la fuente!",
        "Tip de ahorro: el modo suspensión de Windows gasta mucha energía. Apaga bien el equipo si ya acabaste.",
        "Tu computadora registra consumo vampiro. Desenchufa el multicontacto para un ahorro absoluto.",
        "Cortemos el gasto invisible. Mantener la PC conectada sin usarla suma varios pesos a tu recibo.",
        "Los periféricos y monitores en reposo gastan energía constante. Quítales la corriente ahora.",
        "Tu equipo está inactivo pero sumando watts. Desconecta tu estación para maximizar tu ahorro.",
        "La fuente de poder sigue activa internamente. Bajarle el switch es el mejor hábito ecológico.",
        "Detectamos fuga pasiva en tu setup. Libera tu recibo de este gasto apagando todo desde la regleta.",
        "Un equipo de escritorio desconectado es dinero ahorrado. Evita mantenerla en modo espera.",
        "Haz que tu medidor descanse en la noche desconectando por completo tu computadora."
    ]
}

msg_plancha = {
    "mantenimiento": [
        "El termostato de la plancha parece inestable; calienta a destiempo. Revisa si tiene fallas.",
        "Notamos saltos en la resistencia. El sarro del agua podría estar afectando el sistema interno.",
        "Peligro: picos de corriente graves en tu plancha. Revisa que el recubrimiento del cable esté intacto.",
        "Tu plancha tiene un comportamiento eléctrico errático. Esto podría quemar tus prendas, ten cuidado.",
        "Fluctuaciones térmicas detectadas. La resistencia interna podría estar a punto de romperse.",
        "La energía en tu plancha fluye de forma anormal. Una limpieza de calderín prevendría fallos.",
        "Alerta de seguridad: los saltos de voltaje en un aparato de calor son peligrosos. ¡Revísala!",
        "Tu plancha jala corriente a tropezones. Verifica que el enchufe no esté haciendo falso contacto.",
        "El sistema de vapor y calor muestra inestabilidad constante. Podría requerir un reemplazo pronto.",
        "Cuidado con la plancha. La variación eléctrica sugiere desgaste severo en sus componentes."
    ],
    "uso": [
        "Peligro por descuido: tu plancha lleva demasiado tiempo conectada al máximo de temperatura.",
        "El tiempo de uso excede lo normal para una plancha. ¡Apágala de inmediato si te alejaste de ella!",
        "Riesgo inminente de incendio. La resistencia lleva demasiado tiempo activa de forma sostenida.",
        "Le estás exigiendo el máximo a la plancha por horas. El plástico del asa podría derretirse.",
        "Tu plancha está generando calor crítico continuo. Pausa el planchado para que baje su estrés térmico.",
        "Sesión sumamente larga. Cuidar tu plancha apagándola unos minutos prolongará su vida útil.",
        "El calor acumulado es altísimo. Reducir la temperatura o apagarla prevendrá un accidente en casa.",
        "Notamos un uso ininterrumpido inusual. Las planchas caseras necesitan pausas de enfriamiento.",
        "Tu equipo está al límite. Apagarla un rato protegerá tanto a la ropa como al electrodoméstico.",
        "Advertencia de uso excesivo. La plancha necesita un respiro, ¡no la dejes encendida y desatendida!"
    ],
    "ahorro": [
        "Si ya terminaste de planchar, ¡tu plancha sigue conectada! Quítala de la pared por puro ahorro.",
        "Tip de seguridad y ahorro: desenchufa tu plancha en cuanto acabes la última prenda.",
        "Las planchas gastan muchísima luz en modo pasivo. Desconéctala para detener tu medidor de inmediato.",
        "Aprovecha el calor residual desenchufando la plancha antes de terminar tus últimas prendas ligeras.",
        "Tu plancha registra un fuerte consumo vampiro. Evita ese gasto sacándola de la clavija.",
        "Evita consumos silenciosos y riesgos. Una plancha desconectada es la mejor práctica del hogar.",
        "Tu equipo sigue sumando watts a tu recibo. Si la tabla de planchar ya está libre, ¡desenchúfalo!",
        "Haz que tu cuenta de luz baje evitando dejar electrodomésticos de calor conectados sin uso.",
        "Detectamos fuga pasiva. Tu plancha está robando energía que podrías estar ahorrando hoy.",
        "El modo de espera en aparatos de calor no existe. Desconecta tu plancha al 100% para ahorrar."
    ]
}

msg_cafetera = {
    "mantenimiento": [
        "El calentador de agua presenta variaciones. Hacerle una limpieza profunda de sarro le ayudaría mucho.",
        "Inestabilidad en la resistencia de la cafetera. El flujo de agua interna podría estar obstruido.",
        "Notamos comportamiento errático. Revisa que el botón de encendido no esté haciendo falso contacto.",
        "Tu cafetera registra saltos de consumo. La placa calefactora podría estar al borde de fallar.",
        "Fluctuaciones eléctricas en tu café de la mañana. Verifica que el cable de corriente esté íntegro.",
        "Alerta preventiva: las variaciones en la cafetera indican un esfuerzo térmico inusual. ¡Revísala!",
        "Tu electrodoméstico jala corriente de forma atípica. Un chequeo evitará que te quedes sin café.",
        "El desgaste de la placa base es evidente por los picos registrados. Mantén vigilancia al usarla.",
        "Comportamiento inestable constante. Descalcificar tu cafetera es el mejor mantenimiento hoy.",
        "Cuidado con los componentes internos, los saltos de energía sugieren un cortocircuito en gestación."
    ],
    "uso": [
        "Tu cafetera lleva horas manteniendo la jarra caliente. ¡El café ya debe estar quemado, apágala!",
        "Uso muy prolongado detectado. Apaga la cafetera para que no se desgaste inútilmente su placa base.",
        "La resistencia lleva mucho tiempo forzada. ¿Preparaste para un batallón o se te olvidó encendida?",
        "El calor continuo bajo la jarra está fatigando tu cafetera. Dale un descanso desenchufándola.",
        "Has mantenido la cafetera al máximo por demasiado tiempo. El estrés térmico podría quebrar el cristal.",
        "Notamos un uso intensivo que supera lo normal. Pausar el calentador protegerá tu equipo.",
        "El esfuerzo prolongado acorta la vida útil de tu cafetera. ¡Déjala enfriarse por un buen rato!",
        "Cuidemos tu máquina de café. Apagarla tras servir la primera taza es el mejor hábito posible.",
        "El nivel de exigencia es alto y sostenido. Reducir el tiempo de encendido prevendrá averías internas.",
        "Tu cafetera requiere enfriarse. Si el café ya está listo, suspende su uso para cuidar sus resistencias."
    ],
    "ahorro": [
        "La maquinita sigue en modo espera o marcando la hora. ¡Desenchúfala para ahorrar de verdad!",
        "Tip de ahorro: si ya tomaste tu café, no dejes el aparato conectado a la red sumando energía.",
        "Elimina el consumo pasivo de tu cocina desconectando la cafetera desde la clavija de la pared.",
        "Tu cafetera está inactiva pero su pantallita sigue gastando luz. ¡Quítale la corriente y ahorra!",
        "Evita el consumo vampiro de los aparatos de cocina. Una cafetera desconectada cuida tu bolsillo.",
        "Registramos fuga fantasma en tu cafetera. Cada hora conectada es un centavo que se va sin sentido.",
        "El indicador LED de tu cafetera suma a tu recibo. Desenchufarla es un excelente consejo de ahorro.",
        "Si el desayuno terminó, también debería terminar el consumo eléctrico. Desconecta tu máquina.",
        "Detectamos energía fluyendo a la cafetera apagada. Corta ese desperdicio desconectándola ya.",
        "Logra la máxima eficiencia en tu hogar dejando la cafetera desenchufada hasta el día de mañana."
    ]
}

msg_consola = {
    "mantenimiento": [
        "Tu consola registra inestabilidad eléctrica. Cuidado, un pico alto podría corromper el disco duro.",
        "Hay anomalías en la corriente de la consola. Verifica que tu regleta esté en perfecto estado.",
        "Variaciones de voltaje detectadas. Un apagón en este momento dañaría tus archivos de guardado.",
        "La fuente de poder interna muestra desgaste. Conectar tu consola a un supresor de picos es urgente.",
        "Notamos ruido eléctrico en tu sistema de juegos. Asegúrate de que el cable principal encaje bien.",
        "Comportamiento errático en la consola. Revisa que sus salidas de aire estén libres de pelusa.",
        "Alerta de hardware: los saltos de consumo indican que el procesador está recibiendo energía sucia.",
        "La inestabilidad eléctrica persistente acortará la vida de tu consola. Un regulador es la solución.",
        "Detectamos picos de tensión sospechosos. Protege tu equipo para no freír la placa madre.",
        "Tu consola está sufriendo fallas de energía. No actualices el sistema hasta estabilizar la corriente."
    ],
    "uso": [
        "Sesión de juego muy larga. La consola necesita ventilarse para no sobrecalentar el procesador.",
        "El sistema está a tope de capacidad. Asegúrate de que tu consola tenga espacio libre para respirar.",
        "Los ventiladores internos deben estar girando al máximo. ¡Apaga un rato la consola para cuidarla!",
        "Maratón detectado. Pausar el juego unos 20 minutos evitará que los componentes térmicos se degraden.",
        "Tu consola lleva horas renderizando al máximo. El estrés térmico podría secar la pasta térmica.",
        "Le estás exigiendo todo a tu consola de videojuegos hoy. Un respiro extenderá sus años de vida.",
        "Notamos un uso intensivo y sostenido. Si la carcasa se siente muy caliente, apágala de inmediato.",
        "Cuidemos tu centro de entretenimiento. El uso extremo continuo es la principal causa de fallas gráficas.",
        "La carga de procesamiento está al límite. Cerrar aplicaciones en segundo plano podría ayudar.",
        "Tu sistema requiere enfriarse urgentemente. Termina tu partida y dale un descanso a la consola."
    ],
    "ahorro": [
        "El modo reposo y las descargas en segundo plano gastan muchísima luz. ¡Apágala por completo!",
        "Tip de ahorro: los LEDs y funciones ocultas chupan energía. Si no juegas hoy, desconecta la consola.",
        "Tu consola está inactiva pero conectada a internet. Desenchufarla evitará ese enorme gasto pasivo.",
        "Evita que tu consola descargue cosas de noche gastando luz inútilmente. ¡Desconéctala de la pared!",
        "Tu sistema registra un consumo vampiro altísimo. Apagarla del botón no basta, quita el cable.",
        "El modo de encendido rápido te cuesta dinero mensual. Apaga bien la consola y nota la diferencia.",
        "Si ya terminaste de jugar, desconectar la consola es un hábito gamer que cuida tu economía.",
        "Detectamos fuga fantasma. Tu consola sigue consumiendo energía para mantenerse lista. Desenchúfala.",
        "Ahorro extremo: conecta tu consola a un multicontacto con interruptor y apágalo cuando no la uses.",
        "No dejes que tu consola sume pesos a tu recibo eléctrico mientras duermes. Libera la corriente."
    ]
}

msg_secadora = {
    "mantenimiento": [
        "El motor de la secadora sufre picos de esfuerzo. Revisa que el filtro trasero no esté tapado de pelusas.",
        "Comportamiento anómalo en el ventilador interno. Podría estar a punto de atascarse por el cabello.",
        "Fluctuación en la resistencia térmica. La secadora podría aventar chispas si no se limpia pronto.",
        "Inestabilidad eléctrica grave detectada. El motor interno de tu secadora tiene mucha fricción.",
        "Tu secadora de cabello registra variaciones. Verifica que el cable no esté enrollado y trozado por dentro.",
        "Alerta de seguridad: los saltos de corriente en un aparato manual de calor son peligrosos. Revísalo.",
        "El consumo de tu secadora es errático. Revisa el interruptor de encendido por si hay un falso contacto.",
        "Notamos picos peligrosos en el consumo térmico. Una revisión de la resistencia evitará quemaduras.",
        "La energía no fluye bien. Limpia la rejilla de succión de la secadora para evitar que se asfixie el motor.",
        "Cuidado, la inestabilidad de la secadora indica un desgaste muy avanzado en sus plásticos internos."
    ],
    "uso": [
        "Uso extendido a máxima potencia. La carcasa debe estar al rojo vivo, apágala por tu propia precaución.",
        "El motor lleva operando mucho tiempo continuo. Existe el riesgo inminente de que el plástico se derrita.",
        "Peligro térmico detectado. Haz una pausa para dejar que la secadora baje su temperatura extrema.",
        "Le estás exigiendo más de la cuenta a la secadora de pelo. Un respiro evitará que se queme el motor.",
        "Tu aparato está bajo un gran estrés térmico y eléctrico. Pausar su uso prolongará su vida muchísimo.",
        "Sesión sumamente larga. Las secadoras caseras no están diseñadas para operar sin pausas.",
        "La resistencia de tu secadora está al límite. Apágarla unos minutos evitará que se funda internamente.",
        "Notamos un uso intensivo que supera el promedio. Deja que tu secadora se enfríe de forma natural.",
        "El esfuerzo continuo sobre el calentador de aire daña los componentes. Cuidala dándole un respiro.",
        "Tu secadora necesita bajar su temperatura urgentemente. Suspende el secado antes de que haya humo."
    ],
    "ahorro": [
        "Nunca dejes la secadora conectada en el baño después de usarla. ¡Desenchúfala por seguridad y ahorro!",
        "Tip clave: por seguridad con la humedad y para evitar consumos pasivos, quítala de la corriente ya.",
        "El aparato está inactivo pero cerrando un circuito. ¡Jálala del enchufe y ahorra energía en casa!",
        "Las secadoras gastan luz silenciosamente si las dejas conectadas. Desenchufarla toma un segundo.",
        "Evita accidentes en el baño y consumos fantasma asegurándote de desconectar totalmente la secadora.",
        "Tu secadora registra pequeña fuga de energía pasiva. Corta el flujo eléctrico para optimizar tu recibo.",
        "Cuidemos tu economía y seguridad. Un electrodoméstico de calor no debe quedarse en espera en el baño.",
        "Ahorro inteligente: desenchufa los equipos de cuidado personal como tu secadora cuando no los uses.",
        "Detectamos energía circulando por el cable de tu secadora apagada. Quítala de la pared para evitarlo.",
        "Haz que tu medidor no registre fugas extra. Mantén tu secadora desconectada y guardada."
    ]
}

# diccionario maestro que asigna explícitamente todos los alias
mensajes_especificos = {
    "cafetera": msg_cafetera,
    "licuadora": msg_licuadora,
    "microondas": msg_microondas,
    "horno de microondas": msg_microondas,
    "horno microondas": msg_microondas,
    "refrigerador": msg_refrigerador,
    "tostador pan": msg_tostador,
    "tostadora": msg_tostador,
    "plancha ropa": msg_plancha,
    "plancha de ropa": msg_plancha,
    "plancha": msg_plancha,
    "secadora pelo": msg_secadora,
    "secadora de pelo": msg_secadora,
    "secadora": msg_secadora,
    "computadora": msg_pc,
    "computadora escritorio": msg_pc,
    "computadora de escritorio": msg_pc,
    "laptop": msg_laptop,
    "laptop (cargando)": msg_laptop,
    "tv led": msg_tv,
    "tv": msg_tv,
    "consola videojuegos": msg_consola,
    "consola de videojuegos": msg_consola,
    "consola": msg_consola,
    "calefactor portatil": msg_calefactor,
    "calefactor": msg_calefactor,
    "ventilador de pedestal": msg_ventilador,
    "ventilador pedestal": msg_ventilador,
    "ventilador": msg_ventilador
}

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

    # obtenemos los mensajes del diccionario maestro sin mapeos extras
    mensajes_del_aparato = mensajes_especificos.get(device_type, {})

    # regla a mantenimiento e inestabilidad
    if is_anomaly_ia:
        stats['conteo_anomalias'] += 1
    else:
        if stats['conteo_anomalias'] > 0:
            stats['conteo_anomalias'] -= 1 
    
    # si acumula 10 errores recientes
    if stats['conteo_anomalias'] >= 10:
        mensajes_generales_mantenimiento = [
            f"El dispositivo {device_type} ha presentado comportamiento inestable frecuente recientemente.",
            f"Hemos notado que tu {device_type} ha tenido varias fluctuaciones extrañas. Seria bueno revisarlo.",
            f"Atencion: el {device_type} registro multiples anomalias seguidas. Sugerimos un chequeo preventivo."
        ]
        
        lista_final = mensajes_del_aparato.get("mantenimiento", mensajes_generales_mantenimiento)

        recomendacion = {
            "titulo": "Revision Recomendada",
            "mensaje": random.choice(lista_final),
            "tipo": "MANTENIMIENTO", 
            "prioridad": "ALTA"
        }
        stats['conteo_anomalias'] = 0 
        return recomendacion

    # regla b sobrecarga y fatiga
    umbral_fatiga = max_limit * 0.85
    
    if current_watts > umbral_fatiga:
        stats['ciclos_alto_consumo'] += 1
    else:
        stats['ciclos_alto_consumo'] = 0 
    
    # 150 ciclos son aprox 5 minutos continuos
    if stats['ciclos_alto_consumo'] > 150:
        mensajes_generales_uso = [
            f"El dispositivo lleva mucho tiempo operando al limite ({int(current_watts)}W). Podria fatigarse.",
            f"Tu {device_type} lleva un buen rato trabajando al maximo. Cuidado con el sobrecalentamiento.",
            f"Precaucion: uso intenso prolongado en el {device_type}. Considera darle un descanso pronto."
        ]
        
        lista_final = mensajes_del_aparato.get("uso", mensajes_generales_uso)

        recomendacion = {
            "titulo": "Posible Sobrecalentamiento",
            "mensaje": random.choice(lista_final),
            "tipo": "USO", 
            "prioridad": "MEDIA"
        }
        stats['ciclos_alto_consumo'] = 0
        return recomendacion

    # regla c consumo vampiro para ahorro dinamico
    if 0.0 < current_watts <= standby_max_limit:
        stats['ciclos_standby'] += 1
    else:
        stats['ciclos_standby'] = 0 
    
    # 250 ciclos son aprox 30 minutos detectando consumo hormiga
    if stats['ciclos_standby'] > 250:
        mensajes_generales_ahorro = [
            f"Tu {device_type} parece estar en espera gastando energia inutilmente. Desconectalo si no lo usas.",
            f"Detectamos que tu {device_type} esta en modo reposo. Desconectalo de la pared para ahorrar energia.",
            f"El {device_type} esta consumiendo energia fantasma. Si ya terminaste de usarlo, apagalo por completo."
        ]
        
        lista_final = mensajes_del_aparato.get("ahorro", mensajes_generales_ahorro)

        recomendacion = {
            "titulo": "Consumo Vampiro Detectado",
            "mensaje": random.choice(lista_final),
            "tipo": "AHORRO", 
            "prioridad": "BAJA"
        }
        stats['ciclos_standby'] = 0 
        return recomendacion

    return None

    return None