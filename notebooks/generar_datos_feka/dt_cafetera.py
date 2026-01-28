import pandas as pd
import random
import string
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from datetime import datetime, timedelta

# 1. credenciales
DB_USER = "postgres"
DB_PASS = "1234"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "myvolt_local"

pass_encoded = quote_plus(DB_PASS)
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{pass_encoded}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ==========================================
# CONFIGURACIÓN: CAFETERA
# ==========================================
NOMBRE_APARATO = "Cafetera (Entrenamiento)" 

# Consumo
WATTS_ENCENDIDO = (800, 1200)
WATTS_APAGADO = (0, 3)     

# Ciclos
MINUTOS_ENCENDIDO = (30, 60)   
MINUTOS_APAGADO = (30, 60)   

CANTIDAD_DATOS = 3000          
PORCENTAJE_ANOMALIAS = 0.02

# ==========================================
# LÓGICA
# ==========================================

def obtener_id_automatico(conn):
    sql_buscar = text("SELECT id_device FROM devices WHERE device_name = :nom")
    res = conn.execute(sql_buscar, {"nom": NOMBRE_APARATO}).fetchone()
    
    if res:
        return res[0]
    else:
        mac_random = "00:00:00:" + ":".join(["{:02x}".format(random.randint(0, 255)) for _ in range(3)])
        print(f"creando dispositivo '{NOMBRE_APARATO}' con mac {mac_random}...")
        
        sql_crear = text("""
            INSERT INTO devices (device_name, device_type, mac_address, status, created_at)
            VALUES (:nom, 'Simulado_ML', :mac, 'active', NOW())
            RETURNING id_device
        """)
        res = conn.execute(sql_crear, {"nom": NOMBRE_APARATO, "mac": mac_random}).fetchone()
        conn.commit() # Guardamos la creación del dispositivo
        return res[0]

def generar_e_insertar():
    try:
        with engine.connect() as conn:
            device_id = obtener_id_automatico(conn)
            
            print(f"generando {CANTIDAD_DATOS} datos para id: {device_id}...")
            
            data_rows = []
            fecha = datetime.now() - timedelta(minutes=CANTIDAD_DATOS)
            
            esta_encendido = random.choice([True, False])
            contador_ciclo = 0
            limite_ciclo = random.randint(*MINUTOS_ENCENDIDO) if esta_encendido else random.randint(*MINUTOS_APAGADO)
            
            for _ in range(CANTIDAD_DATOS):
                fecha += timedelta(minutes=1)
                contador_ciclo += 1
                
                if contador_ciclo >= limite_ciclo:
                    esta_encendido = not esta_encendido
                    contador_ciclo = 0
                    if esta_encendido:
                        limite_ciclo = random.randint(*MINUTOS_ENCENDIDO)
                    else:
                        limite_ciclo = random.randint(*MINUTOS_APAGADO)
                
                rango = WATTS_ENCENDIDO if esta_encendido else WATTS_APAGADO
                watts = random.uniform(*rango)
                
                if random.random() < PORCENTAJE_ANOMALIAS:
                    tipo = random.choice(["pico", "caida"])
                    if tipo == "pico":
                        watts = watts * 3
                    elif tipo == "caida" and esta_encendido:
                        watts = 0 
                
                data_rows.append({
                    "id_device": device_id,
                    "time": fecha,
                    "power_w": round(watts, 2),
                    "voltage": 120.0,
                    "current_a": 0.0,
                    "power_factor": 1.0,
                    "apparent_power": 0.0
                })
            
            df = pd.DataFrame(data_rows)
            df.to_sql('consumption_data', con=conn, if_exists='append', index=False)
            
            # --- LA LÍNEA MÁGICA QUE FALTABA ---
            conn.commit() 
            # -----------------------------------
            
            print("¡listo! datos inyectados CORRECTAMENTE en la tabla consumption_data.")
            
    except Exception as e:
        print(f"ERROR DETALLADO: {e}")

if __name__ == "__main__":
    generar_e_insertar()