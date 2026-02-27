# usamos la imagen base compatible con azure y procesadores amd o intel
FROM --platform=linux/amd64 python:3.10-slim

# configuraciones de python para no escribir archivos cache y mostrar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# creamos la carpeta de trabajo
WORKDIR /code

# copiamos los requerimientos y los instalamos
COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copiamos todo el codigo del proyecto
COPY . /code/

# exponemos el puerto para comunicacion
EXPOSE 8000

# comando de arranque
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
