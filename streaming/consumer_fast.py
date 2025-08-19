import json
import requests
from kafka import KafkaConsumer

# API_URL = "http://localhost:8800/predict" # TODO: Deprecated - Eliminar esta linea en el futuro | Apunta a FastAPI
API_URL = "http://localhost:32001" # Apunta a la función de Nuclio


# Configuración del consumidor Kafka
consumer = KafkaConsumer(
    'appartment_data',
    bootstrap_servers=['localhost:9094'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='ml-processing-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Escuchando mensajes en el tópico 'appartment_data'...")

try:
    for message in consumer:
        data = message.value
        print(f"Received from Kafka: {data}")

        try:
            response = requests.post(API_URL, json=data)
            response.raise_for_status()
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