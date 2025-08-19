import json, time, pickle, os
import boto3
import mlflow
from pydantic import BaseModel, ValidationError
import pandas as pd


class ModelInput(BaseModel):
    expenses_amount: float
    total_mts: float
    covered_mts: float
    rooms: int
    bedrooms: int
    bathrooms: int
    garages: int
    antique: int


def _load_model(context, model_name: str, alias: str) -> tuple:
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://s3:9000"

    model_ml = None
    version_model_ml = None
    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri("http://mlflow:5005")
        client_mlflow = mlflow.MlflowClient()
        context.logger.info(f"Datos de conexion a MLflow: {client_mlflow}")

        context.logger.info(f"Intentando cargar modelo: {model_name} con alias: {alias}")
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)

        context.logger.info(f"Datos del modelo MLflow: {model_data_mlflow}")
        if model_data_mlflow is None:
            context.logger.warning(f"No se encontró el modelo '{model_name}' con alias '{alias}' en MLflow.")
            return None, None, {}
        context.logger.info(f"Ruta del modelo en MLflow: {model_data_mlflow.source}")
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
        context.logger.info(f"Modelo cargado exitosamente: Versión {version_model_ml}")
    except mlflow.MlflowException as e:
        context.logger.error(f"Error al conectar o cargar el modelo desde MLflow: {e}")
        return None, None, {}
    except Exception as e:
        context.logger.error(f"Error al conectar o cargar el modelo desde MLflow: {e}")
        return None, None, {}

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url="http://s3:9000",
            aws_access_key_id="minio",
            aws_secret_access_key="minio123",
            region_name="us-east-1",
        )
        bucket_name = "data"
        scaler_filename = "/scalers/scaler_X.pkl"
        response = s3.get_object(Bucket=bucket_name, Key=scaler_filename)
        scaler_data = response["Body"].read()

        # Deserializar el objeto StandardScaler
        scaler_X = pickle.loads(scaler_data)

        scaler_filename = "/scalers/scaler_y.pkl"
        response = s3.get_object(Bucket=bucket_name, Key=scaler_filename)
        scaler_data = response["Body"].read()

        # Deserializar el objeto StandardScaler
        scaler_y = pickle.loads(scaler_data)

        data_dictionary = {"scaler_X": scaler_X, "scaler_y": scaler_y}
        context.logger.info("Scalers cargados correctamente desde S3.")
    except Exception as e:
        context.logger.warning(f"Informacion de estandarizado no encontrada: {e}")
        data_dictionary = {}

    return model_ml, version_model_ml, data_dictionary


def _get_or_load_model(context):
    """Carga el modelo solo si no está ya cargado (patrón singleton)"""
    # Verificar si ya tenemos el modelo cargado
    if hasattr(context, "_cached_model") and context._cached_model is not None:
        context.logger.info("Usando modelo desde caché")
        return context._cached_model, context._cached_data_dict

    context.logger.info("Cargando modelo por primera vez...")
    model, version_model, data_dict = _load_model(context, "precio_propiedades_model_prod", "prod")

    if model is None:
        context.logger.error("No se pudo cargar el modelo")
        return None, None

    # Guardar en el contexto para futuras invocaciones
    context._cached_model = model
    context._cached_data_dict = data_dict
    context._cached_version = version_model

    context.logger.info(f"Modelo cargado y guardado en caché. Versión: {version_model}")
    return model, data_dict


def _handle_post(context, event):
    try:
        # Cargar modelo si es necesario (lazy loading)
        model, data_dict = _get_or_load_model(context)

        if model is None:
            return context.Response(
                body="Error: Modelo no disponible. MLflow podría no estar listo o no se ha entrenado un modelo.",
                headers={},
                content_type="text/plain",
                status_code=503,  # Service Unavailable
            )

        # El body viene como bytes -> decodificamos
        data = event.body

        # Validamos con Pydantic
        input_data = ModelInput(**data)

        start_time = time.time()
        df = pd.DataFrame([input_data.model_dump()])
        scaler_X = data_dict["scaler_X"]
        scaler_y = data_dict["scaler_y"]
        df = scaler_X.transform(df)
        prediction = model.predict(df)
        unstandarize_prediction = float(scaler_y.inverse_transform(prediction.reshape(-1, 1))[0][0])
        execution_time = time.time() - start_time
        body = {
            "prediction": round(unstandarize_prediction, 2),
            "execution_time": execution_time,
        }
        return context.Response(
            body=json.dumps(body),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except ValidationError as e:
        # Error de validación de Pydantic
        return context.Response(body=str(e), headers={}, content_type="application/json", status_code=400)

    except Exception as e:
        # Cualquier otro error
        return context.Response(body=str(e), headers={}, content_type="text/plain", status_code=500)


def _handle_post_mock(context, event):
    # Simulación de una predicción
    try:
        body = {
            "prediction": 123456.78,
            "execution_time": 0.123,
        }
        return context.Response(
            body=json.dumps(body),
            headers={},
            content_type="application/json",
            status_code=200,
        )
    except Exception as e:
        return context.Response(body=str(e), headers={}, content_type="text/plain", status_code=500)


def predict(context, event):
    method = event.method
    context.logger.info(f"HTTP method: {method}")

    match method:
        case "POST":
            return _handle_post(context, event)
        case "GET":
            return context.Response(
                body="Esta todo ok. Ejecuta un POST para predecir.",
                headers={},
                content_type="text/plain",
                status_code=200,
            )
        case _:
            return context.Response(
                body="Method not allowed",
                headers={},
                content_type="text/plain",
                status_code=405,
            )
