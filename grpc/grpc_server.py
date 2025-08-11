import grpc
from concurrent import futures
import time
import pickle
import boto3
import mlflow
from mlflow.exceptions import MlflowException
import pandas as pd
import numpy as np
import logging

import predict_pb2
import predict_pb2_grpc


from botocore.client import Config
import os

# Configuración de logging para consola Docker
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

# Copia la función load_model del fastapi_app.py

def load_model(model_name: str, alias: str):
    logger.info(f"[load_model] Iniciando carga del modelo: {model_name} con alias: {alias}")
    model_ml = None
    version_model_ml = None
    try:
        mlflow.set_tracking_uri('http://mlflow:5005')
        client_mlflow = mlflow.MlflowClient()
        logger.info(f"[load_model] Intentando obtener versión del modelo desde MLflow")
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        logger.info(f"[load_model] Datos del modelo MLflow: {model_data_mlflow}")
        if model_data_mlflow is None:
            logger.warning(f"[load_model] No se encontró el modelo '{model_name}' con alias '{alias}' en MLflow.")
            return None, None, {}
        logger.info(f"[load_model] Ruta del modelo en MLflow: {model_data_mlflow.source}")
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
        logger.info(f'[load_model] Modelo cargado exitosamente: Versión {version_model_ml}')
    except MlflowException as e:
        logger.error(f'[load_model] Error al conectar o cargar el modelo desde MLflow: {e}')
    except Exception as e:
        logger.error(f'[load_model] Error inesperado al cargar el modelo: {e}')

    try:
        logger.info('[load_model] Intentando cargar scalers desde S3...')
        s3 = boto3.client(
                's3',
                endpoint_url=os.environ.get('AWS_ENDPOINT_URL_S3'),  # Usa la variable de entorno configurada en docker-compose
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                config=Config(signature_version='s3v4'),
                region_name='auto'
            )
        
        bucket_name = os.environ.get('DATA_REPO_BUCKET_NAME')
        scaler_filename = 'scalers/scaler_X.pkl'
        logger.info(f'[load_model] Descargando scaler_X desde S3: bucket={bucket_name}, key={scaler_filename}')
        response = s3.get_object(Bucket=bucket_name, Key=scaler_filename)
        scaler_data = response['Body'].read()
        scaler_X = pickle.loads(scaler_data)
        logger.info('[load_model] scaler_X cargado correctamente.')
        scaler_filename = 'scalers/scaler_y.pkl'
        logger.info(f'[load_model] Descargando scaler_y desde S3: bucket={bucket_name}, key={scaler_filename}')
        response = s3.get_object(Bucket=bucket_name, Key=scaler_filename)
        scaler_data = response['Body'].read()
        scaler_y = pickle.loads(scaler_data)
        logger.info('[load_model] scaler_y cargado correctamente.')
        data_dictionary = {'scaler_X': scaler_X, 'scaler_y': scaler_y}
        logger.info('[load_model] Scalers cargados correctamente desde S3.')
    except Exception as e:
        logger.warning(f'[load_model] Informacion de estandarizado no encontrada: {e}')
        data_dictionary = {}
    logger.info(f'[load_model] Retornando modelo, versión y data_dictionary: model_ml={model_ml is not None}, version_model_ml={version_model_ml}, data_dict_keys={list(data_dictionary.keys())}')
    return model_ml, version_model_ml, data_dictionary

class PropertyPricePredictorServicer(predict_pb2_grpc.PropertyPricePredictorServicer):
    def __init__(self):
        logger.info('[PropertyPricePredictorServicer] Inicializando servicer y cargando modelo...')
        self.model, self.version_model, self.data_dict = load_model("precio_propiedades_model_prod", "prod")
        logger.info(f'[PropertyPricePredictorServicer] Modelo cargado: {self.model is not None}, versión: {self.version_model}, data_dict_keys: {list(self.data_dict.keys())}')

    def Predict(self, request, context):
        start_time = time.time()
        logger.info('[PropertyPricePredictorServicer] Inicializando servicer y cargando modelo...')
        self.model, self.version_model, self.data_dict = load_model("precio_propiedades_model_prod", "prod")
        logger.info(f"[Predict] Request recibido: {request}")
        # Convertir los datos de entrada a un DataFrame de pandas
        df = pd.DataFrame([{ 
            'expenses_amount': request.expenses_amount,
            'total_mts': request.total_mts,
            'covered_mts': request.covered_mts,
            'rooms': request.rooms,
            'bedrooms': request.bedrooms,
            'bathrooms': request.bathrooms,
            'garages': request.garages,
            'antique': request.antique
        }])
        logger.info(f"[Predict] DataFrame de entrada: {df}")
        try:
            scaler_X = self.data_dict['scaler_X']
            scaler_y = self.data_dict['scaler_y']
        except KeyError as e:
            logger.error(f"[Predict] No se encontraron los scalers necesarios en data_dict: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('Scalers no encontrados en el servidor.')
            execution_time = time.time() - start_time
            return predict_pb2.PredictResponse(prediction=0.0, execution_time=execution_time)
        try:
            df_scaled = scaler_X.transform(df)
            logger.info(f"[Predict] DataFrame escalado: {df_scaled}")
            prediction = self.model.predict(df_scaled)
            logger.info(f"[Predict] Predicción escalada: {prediction}")
            unstandarize_prediction = float(scaler_y.inverse_transform(prediction.reshape(-1, 1))[0][0])
            logger.info(f"[Predict] Predicción final (desescalada): {unstandarize_prediction}")
            execution_time = time.time() - start_time
            return predict_pb2.PredictResponse(prediction=round(unstandarize_prediction,2), execution_time=execution_time)
        except Exception as e:
            logger.error(f"[Predict] Error durante la predicción: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('Error durante la predicción.')
            execution_time = time.time() - start_time
            return predict_pb2.PredictResponse(prediction=0.0, execution_time=execution_time)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    predict_pb2_grpc.add_PropertyPricePredictorServicer_to_server(PropertyPricePredictorServicer(), server)
    server.add_insecure_port('[::]:50051')
    logger.info('[serve] gRPC server running on port 50051...')
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info('[serve] Servidor detenido por KeyboardInterrupt.')
        server.stop(0)

if __name__ == '__main__':
    logger.info('[main] Iniciando servidor gRPC...')
    serve()
