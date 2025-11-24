# 1. Usamos la imagen base compatible con Azure (linux/amd64)
FROM --platform=linux/amd64 python:3.10-slim

# 2. Configuraciones de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Crear carpeta de trabajo
WORKDIR /code

# 4. Copiar requerimientos e instalarlos
COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copiar todo el proyecto
COPY . /code/

# 6. Exponer puerto (informativo)
EXPOSE 8000

# 7. Comando de arranque
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]