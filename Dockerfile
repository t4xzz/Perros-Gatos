# Usar imagen base de Python
FROM python:3.10-slim

# Instalar dependencias del sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Exponer el puerto que usará Cloud Run
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app