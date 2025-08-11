import os
from matplotlib import pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import datetime


print(mlflow.__file__)
print(mlflow.log_artifact.__code__.co_filename)  # Archivo físico con la definición

# Variables de entorno antes de importar mlflow
os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] =  os.environ.get("AWS_SECRET_ACCESS_KEY")

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_S3_SIGNATURE_VERSION"] = "s3v4"

# Para virtual addressing style:
os.environ["AWS_S3_FORCE_PATH_STYLE"] = "false"  # 'false' para virtual addressing
# os.environ["AWS_S3_FORCE_PATH_STYLE"] = "true" # 'true' para path style, según tu endpoint

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.googleapis.com"
os.environ["AWS_ENDPOINT_URL_S3"] = "https://storage.googleapis.com"

os.environ["S3_USE_SIGV4"] = "True"  # Asegura usar firma v4

mlflow.set_tracking_uri('http://localhost:5005')

experiment_name = "precio_propiedades_model_experiment"

if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(name=experiment_name)

experiment = mlflow.get_experiment_by_name(experiment_name)

# Obtener las variables de entorno (puedes ajustarlas según sea necesario)
pg_user = os.getenv('PG_USER', 'airflow')
pg_password = os.getenv('PG_PASSWORD', 'airflow')
pg_host = 'localhost'  # Si estás ejecutando el script desde dentro del contenedor, usa 'postgres'
pg_port = os.getenv('PG_PORT', '5432')
pg_database = os.getenv('PG_DATABASE', 'airflow')

db_url = f'postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}'

# Crear la conexión con SQLAlchemy
engine = create_engine(db_url)

# Leer los datos para entrenar el modelo
query = 'SELECT * FROM public."X_train_scaled";'
X_train = pd.read_sql(query, engine)
X_train[X_train.select_dtypes(['bool']).columns] = X_train.select_dtypes(['bool']).astype(int)

query = 'SELECT * FROM public."X_test_scaled";'
X_test = pd.read_sql(query, engine)
X_test[X_test.select_dtypes(['bool']).columns] = X_test.select_dtypes(['bool']).astype(int)

query = 'SELECT * FROM public."y_train_scaled";'
y_train = pd.read_sql(query, engine)
y_train[y_train.select_dtypes(['bool']).columns] = y_train.select_dtypes(['bool']).astype(int)

query = 'SELECT * FROM public."y_test_scaled";'
y_test = pd.read_sql(query, engine)
y_test[y_test.select_dtypes(['bool']).columns] = y_test.select_dtypes(['bool']).astype(int)



def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    
# Creemos el experimento
experiment_id = get_or_create_experiment("Precio Propiedades")
print(experiment_id)

run_name_parent = "best_hyperparam_"  + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')

xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, enable_categorical=True)
print(X_train.info())
xgb_regressor.fit(X_train, y_train)

artifact_path='xgb_model'
model_uri = ""
model_version = 1

with mlflow.start_run(experiment_id = experiment.experiment_id):
    # Se registran los mejores hiperparámetros
    mlflow.log_params(xgb_regressor.get_params())
    
    # Se obtiene las predicciones del dataset de evaluación
    y_pred = xgb_regressor.predict(X_test)
    # y_test_inversed = scaler_y.inverse_transform(y_test)
    # y_pred_inversed = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    
    # Se calculan las métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("\nDesempeño en el conjunto de prueba:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
    
    # Y las enviamos a MLFlow
    metrics ={
        'mae': mae,
        'mse': mse, 
        'rmse': rmse,
        'r2': r2
        }
    mlflow.log_metrics(metrics)
    # Obtenemos la importancia de características
    feature_importances = xgb_regressor.feature_importances_
    feature_names = X_train.columns  # Excluir la columna objetivo

    # Ordenamos los índices
    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, 10))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')

    # Guardar el gráfico en un archivo temporal
    plot_filename = 'feature_importances.png'
    plt.savefig(plot_filename)

    # Registrar el gráfico en MLflow
    mlflow.log_figure(plt.gcf(), artifact_file="feature_importances.png")


    # Registramos el modelo y los datos de entrenamiento
    mlflow.sklearn.log_model(
            sk_model=xgb_regressor,
            artifact_path=artifact_path,
            serialization_format='cloudpickle',
            registered_model_name="precio_propiedades_model_dev",
            metadata={"model_data_version": model_version}
            )

   
    # Obtenemos la ubicación del modelo guardado en MLFlow
    model_uri = mlflow.get_artifact_uri(artifact_path)


print('Finalizado')


from mlflow import MlflowClient

client = MlflowClient()
name = "precio_propiedades_model_prod"
desc = "Predice el precio de propiedades en CABA"

# Creamos el modelo productivo
# Chequear si el modelo ya está registrado
registered_models = client.search_registered_models(f"name='{name}'")

if not registered_models:
    client.create_registered_model(name=name, description=desc)



# Guardamos como tag los hiper-parametros en la version del modelo
tags = xgb_regressor.get_params()
tags["model"] = type(xgb_regressor).__name__
tags["mae"] = mae
tags["mse"] = mse
tags["rmse"] = rmse
tags["r2"] = r2

# Guardamos la version del modelo
result = client.create_model_version(
    name=name,
    source=model_uri,
    run_id=model_uri.split("/")[-3],
    tags=tags
)

# Y creamos como la version con el alias de prod para poder levantarlo en nuestro
# proceso de servicio del modelo on-line.
client.set_registered_model_alias(name, "prod", result.version)