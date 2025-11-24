import requests
import json

# La URL de tu API
url = "https://myvolt-api-iachavira.azurewebsites.net/predict/anomaly"

# CASO 1: Datos Normales (TV encendida viendo una peli - rango 50-150W)
datos_normales = {
    "device_type": "TV LED",
    "power_w": 28.5,  
    "voltage": 120.0,
    "current_a": 0.7
}

# CASO 2: Datos Anómalos (Un pico de energía absurdo)
datos_anomalos = {
    "device_type": "TV LED",
    "power_w": 450.0, # 450W es demasiado para una TV LED normal
    "voltage": 120.0,
    "current_a": 3.75
}

print("--- Enviando Caso Normal (85W) ---")
try:
    response = requests.post(url, json=datos_normales)
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"No se pudo conectar: {e}")

print("\n--- Enviando Caso Anómalo (450W) ---")
try:
    response = requests.post(url, json=datos_anomalos)
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"No se pudo conectar: {e}")