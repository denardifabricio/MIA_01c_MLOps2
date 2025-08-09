# Usa la imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .
COPY predict.proto .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Genera los archivos de Python a partir del proto
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. predict.proto

# Copia el resto del código de la aplicación a /app
COPY . .

# Expone el puerto 8501 para Streamlit
EXPOSE 8501

# Comando por defecto para ejecutar Streamlit
CMD ["streamlit", "run", "streamlit_app.py"]
