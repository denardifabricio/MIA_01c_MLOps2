#writefile hands_on_streaming/consumer.py

import sys, os

# Agregar la carpeta grpc (la que contiene predict_pb2.py) al inicio del sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../grpc')))
import predict_pb2
import predict_pb2_grpc


import json
import requests
from kafka import KafkaConsumer
import pandas as pd
import numpy as np
import grpc
#import predict_pb2
#import predict_pb2_grpc



# Crear canal y stub gRPC
channel = grpc.insecure_channel('localhost:50051')  # puerto donde corre el servidor gRPC
stub = predict_pb2_grpc.PropertyPricePredictorStub(channel)


# Configuración del consumidor Kafka
consumer = KafkaConsumer(
    'appartment_data',
    bootstrap_servers=['localhost:9094'], # Usar el puerto expuesto por Docker Compose
    auto_offset_reset='earliest', # Empieza a leer desde el principio si no hay offset guardado
    enable_auto_commit=True,
    group_id='ml-processing-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Escuchando mensajes en el tópico 'appartment_data'...")

try:
    for message in consumer:
        data = message.value
        print(f"Received from Kafka: {data}")

        # Crear request gRPC usando el mensaje de Kafka
        request = predict_pb2.PredictRequest(
            expenses_amount=float(data.get("expenses_amount", 0)),
            total_mts=float(data.get("total_mts", 0)),
            covered_mts=float(data.get("covered_mts", 0)),
            rooms=int(data.get("rooms", 0)),
            bedrooms=int(data.get("bedrooms", 0)),
            bathrooms=int(data.get("bathrooms", 0)),
            garages=int(data.get("garages", 0)),
            antique=int(data.get("antique", 0)),
        )

        # Llamar al servicio gRPC
        try:
            response = stub.Predict(request)
            print(f"Prediction from gRPC: {response.prediction}")
        except grpc.RpcError as e:
            print(f"Error al llamar al servidor gRPC: {e}")

        print("-" * 30)

except KeyboardInterrupt:
    print("Deteniendo consumidor...")
finally:
    consumer.close()
    print("Consumidor cerrado.")