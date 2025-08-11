import pickle
import boto3
import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.base import BaseEstimator
import strawberry
from typing import Optional
import logging
import time

from botocore.client import Config
import os


# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_name: str, alias: str):
    model_ml = None
    version_model_ml = None
    try:
        mlflow.set_tracking_uri('http://mlflow:5005')
        client_mlflow = mlflow.MlflowClient()
        logger.info(f"Intentando cargar modelo: {model_name} con alias: {alias}")
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        logger.info(f"Datos del modelo MLflow: {model_data_mlflow}")
        if model_data_mlflow is None:
            logger.warning(f"No se encontró el modelo '{model_name}' con alias '{alias}' en MLflow.")
            return None, None, {}
        logger.info(f"Ruta del modelo en MLflow: {model_data_mlflow.source}")
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
        logger.info(f'Modelo cargado exitosamente: Versión {version_model_ml}')
    except MlflowException as e:
        logger.error(f'Error al conectar o cargar el modelo desde MLflow: {e}')
    except Exception as e:
        logger.error(f'Error al conectar o cargar el modelo desde MLflow: {e}')

    try:
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
        response = s3.get_object(Bucket=bucket_name, Key=scaler_filename)
        scaler_data = response['Body'].read()
        scaler_X = pickle.loads(scaler_data)
        scaler_filename = 'scalers/scaler_y.pkl'
        response = s3.get_object(Bucket=bucket_name, Key=scaler_filename)
        scaler_data = response['Body'].read()
        scaler_y = pickle.loads(scaler_data)
        data_dictionary = {
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
        logger.info('Scalers cargados correctamente desde S3.')
    except Exception as e:
        logger.warning(f'Informacion de estandarizado no encontrada: {e}')
        data_dictionary = {}
    return model_ml, version_model_ml, data_dictionary

# Definir el modelo de entrada para Strawberry
@strawberry.type
class PredictionResult:
    prediction: float
    execution_time: float

@strawberry.input
class PredictInput:
    expensesAmount: float
    totalMts: float
    coveredMts: float
    rooms: int
    bedrooms: int
    bathrooms: int
    garages: int
    antique: int

@strawberry.type
class Query:
    @strawberry.field
    def hello(self) -> str:
        return "Bienvenidos a la API GraphQL para predecir el precio de las propiedades de CABA"

@strawberry.type
class Mutation:
    @strawberry.mutation
    def predict(self, input_data: PredictInput) -> PredictionResult:
        start_time = time.time()
        logger.info(f"Solicitud de predicción recibida: {input_data}")
        model, version_model, data_dict = load_model("precio_propiedades_model_prod", "prod")
        if not model or not data_dict:
            logger.error("No se pudo cargar el modelo o los scalers. Retornando -1.")
            execution_time = time.time() - start_time
            return PredictionResult(prediction=-1, execution_time=execution_time)
        df = pd.DataFrame([{ 
            'expenses_amount': input_data.expensesAmount,
            'total_mts': input_data.totalMts,
            'covered_mts': input_data.coveredMts,
            'rooms': input_data.rooms,
            'bedrooms': input_data.bedrooms,
            'bathrooms': input_data.bathrooms,
            'garages': input_data.garages,
            'antique': input_data.antique
        }])
        logger.info(f"Datos de entrada convertidos a DataFrame: {df}")
        scaler_X = data_dict['scaler_X']
        scaler_y = data_dict['scaler_y']
        df_scaled = scaler_X.transform(df)
        logger.info(f"Datos de entrada escalados: {df_scaled}")
        prediction = model.predict(df_scaled)
        logger.info(f"Predicción estandarizada: {prediction}")
        unstandarize_prediction = float(scaler_y.inverse_transform(prediction.reshape(-1, 1))[0][0])
        logger.info(f"Predicción final desescalada: {unstandarize_prediction}")
        execution_time = time.time() - start_time
        return PredictionResult(prediction=round(unstandarize_prediction, 2), execution_time=execution_time)

schema = strawberry.Schema(query=Query, mutation=Mutation)

# Integración con FastAPI para servir GraphQL
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
logger.info("Inicializando la aplicación FastAPI y configurando el endpoint de GraphQL.")
app = FastAPI()
app.include_router(GraphQLRouter(schema), prefix="/graphql")
