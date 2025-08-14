#%% writefile hands_on_streaming/producer.py
import time
import json
import random
from kafka import KafkaProducer

def serialize_json(obj):
    return json.dumps(obj).encode('utf-8')

# Configuración del productor Kafka
producer = KafkaProducer(
    bootstrap_servers=['localhost:9094'], # Usar el puerto expuesto por Docker Compose
    value_serializer=serialize_json
)

topic_name = 'appartment_data'

print(f"Enviando datos al tópico: {topic_name}")

try:
    for i in range(100):
        expenses_amount = round(random.uniform(80000.0, 300000.0), 2)
        total_mts = random.randint(30, 150)
        covered_mts = total_mts - random.randint(0, 20)
        rooms = random.randint(1, 6)
        bedrooms = random.randint(1, 4)
        bathrooms = random.randint(1, 3)
        garages = random.randint(1, 3)
        antique = random.randint(1, 50)


        data = {
            'expenses_amount': expenses_amount,
            'total_mts': total_mts,
            'covered_mts': covered_mts,
            'rooms': rooms,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'garages': garages,
            'antique': antique

        }

        producer.send(topic_name, value=data)
        print(f"Sent: {data}")
        time.sleep(1) # Enviar un evento cada segundo

except KeyboardInterrupt:
    print("Deteniendo productor...")
finally:
    producer.close()
    print("Productor cerrado.")