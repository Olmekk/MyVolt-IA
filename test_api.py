import requests
import json

# url local de tu api asegúrate de que el puerto coincida con tu docker
# url = "http://127.0.0.1:8000/predict/anomaly"
url = "http://100.100.217.46:8000/predict/anomaly"

# encabezados necesarios para la conexion
headers = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true" 
}

# caso 1 datos normales dentro del limite de la tv led
datos_normales = {
  "device_type": "TV LED",
  "power_w": 60.0,
  "voltage": 120.0,
  "current_a": 0.45
}

print("enviando caso normal de 13.5w")
try:
    response = requests.post(url, json=datos_normales, headers=headers, timeout=10)
    
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    else:
        print(f"error de servidor {response.status_code}: {response.text}")
except Exception as e:
    print(f"no se pudo conectar asegurate de que docker este corriendo: {e}")

print("\nenviando caso de peligro fisico de 450w")
try:
    response = requests.post(url, json=datos_peligro, headers=headers, timeout=10)
    
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    else:
        print(f"error de servidor {response.status_code}: {response.text}")
except Exception as e:
    print(f"no se pudo conectar asegurate de que docker este corriendo: {e}")