import sys, os
import json
from kafka import KafkaConsumer
import pandas as pd
import grpc
from datetime import datetime
import boto3

# --- gRPC setup ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../grpc')))
import predict_pb2
import predict_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = predict_pb2_grpc.PropertyPricePredictorStub(channel)

# # --- S3 (GCS via S3 API) ---
# S3_ENDPOINT = "https://storage.googleapis.com"
# S3_ACCESS_KEY = ""
# S3_SECRET_KEY = ""
# S3_BUCKET_NAME = "mlops2-data1"

# s3 = boto3.client(
#     "s3",
#     endpoint_url=S3_ENDPOINT,
#     aws_access_key_id=S3_ACCESS_KEY,
#     aws_secret_access_key=S3_SECRET_KEY,
#     region_name="us-east-1",
#     config=boto3.session.Config(s3={'addressing_style': 'path'}, signature_version='s3v4')
# )

# --- Kafka consumer ---
consumer = KafkaConsumer(
    "appartment_data",
    bootstrap_servers=["localhost:9094"],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="ml-processing-group",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("Escuchando mensajes en el tópico 'appartment_data'...")

oportunidades = []

try:
    for message in consumer:
        data = message.value
        print(f"Received from Kafka: {data}")

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

        try:
            response = stub.Predict(request)
            predicted_price = response.prediction
            real_price = float(data.get("real_price", 0))
            property_id = data.get("id", "N/A")

            if real_price < predicted_price:
                print(f"OPORTUNIDAD detectada para propiedad {property_id}")
                oportunidades.append({
                    "id": property_id,
                    "precio_real": real_price,
                    "precio_predicho": predicted_price,
                    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                # Guardar Excel temporal
                excel_filename = "./oportunidades.xlsx"
                pd.DataFrame(oportunidades).to_excel(excel_filename, index=False)

                # Subir a GCS vía S3
                # s3.upload_file(excel_filename, S3_BUCKET_NAME, "oportunidades.xlsx")
                # print(f"📤 Archivo subido a s3://{S3_BUCKET_NAME}/oportunidades.xlsx")

        except grpc.RpcError as e:
            print(f"Error al llamar al servidor gRPC: {e}")

except KeyboardInterrupt:
    print("Deteniendo consumidor...")
finally:
    consumer.close()
    print("Consumidor cerrado.")
