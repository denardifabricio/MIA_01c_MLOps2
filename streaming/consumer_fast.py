#writefile hands_on_streaming/consumer.py
import json
import requests
from kafka import KafkaConsumer
import pandas as pd
import numpy as np

# URL de la API FastAPI dentro de Docker
API_URL = "http://localhost:8800/predict"


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

        # Enviar datos a FastAPI
        try:
            response = requests.post(API_URL, json=data)
            if response.status_code == 200:
                prediction = response.json().get("prediction")
                print(f"Prediction from API: {prediction}")
            else:
                print(f"Error al enviar los datos: {response.status_code}")
        except Exception as e:
            print(f"Error al llamar a la API: {e}")

        print("-" * 30)

except KeyboardInterrupt:
    print("Deteniendo consumidor...")
finally:
    consumer.close()
    print("Consumidor cerrado.")