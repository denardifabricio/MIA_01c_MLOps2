FROM python:3.10-slim

WORKDIR /app

# Copia los archivos necesarios
COPY grpc_server.py ./
COPY predict.proto ./
COPY requirements.txt ./

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Genera los archivos de Python a partir del proto
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. predict.proto

# Expone el puerto gRPC
EXPOSE 50051

# Comando para iniciar el servidor
CMD ["python", "grpc_server.py"]
