# usamos la imagen base compatible con azure y procesadores amd o intel
FROM --platform=linux/amd64 python:3.10-slim

# configuraciones de python para no escribir archivos cache y mostrar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# instalacion de dependencias de sistema necesarias
# gcc es para compilar extensiones de python y libpq-dev para la conexion a postgres
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# creamos la carpeta de trabajo
WORKDIR /code

# primero copiamos solo los requerimientos para aprovechar el cache de las capas de docker
COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# copiamos todo el codigo del proyecto (incluyendo las carpetas app y notebooks)
COPY . /code/

# exponemos el puerto para comunicacion
EXPOSE 8000

# comando de arranque usando el path correcto al objeto 'app' dentro de la carpeta 'app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]