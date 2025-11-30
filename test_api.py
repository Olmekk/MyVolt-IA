import requests
import json

# La URL de tu API
# url = "https://unstuttered-hookiest-maryam.ngrok-free.dev/predict/anomaly"
url = "http://127.0.0.1:8005/predict/anomaly"

# Encabezados (Headers) necesarios
headers = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true" # <--- ¡ESTA ES LA LLAVE MAESTRA!
}

# CASO 1: Datos Normales
datos_normales = {
    "device_type": "TV LED",
    "power_w": 400.5,  
    "voltage": 120.0,
    "current_a": 0.7
}

# CASO 2: Datos Anómalos
datos_anomalos = {
    "device_type": "TV LED",
    "power_w": 64.0, 
    "voltage": 120.0,
    "current_a": 0.7
}

print("--- Enviando Caso Normal (13.5W) ---")
try:
    # Agregamos 'headers' y 'timeout'
    response = requests.post(url, json=datos_normales, headers=headers, timeout=10)
    
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"No se pudo conectar (Timeout o Error): {e}")

print("\n--- Enviando Caso Anómalo (450W) ---")
try:
    response = requests.post(url, json=datos_anomalos, headers=headers, timeout=10)
    
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"No se pudo conectar (Timeout o Error): {e}")