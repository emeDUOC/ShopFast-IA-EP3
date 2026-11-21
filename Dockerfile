# Imagen base oficial de Python
FROM python:3.12-slim

# Evita archivos .pyc y asegura logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del proyecto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la API, el modelo y los tests
COPY api/ ./api/
COPY model/ ./model/
COPY tests/ ./tests/

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando de inicio en producción
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]