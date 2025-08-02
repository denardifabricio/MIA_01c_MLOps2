import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


import xgboost as xgb
import mlflow

import boto3
from botocore.client import Config

#pip install psycopg2

os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['ENDPOINT_URL'] = 'http://localhost:9001'
os.environ["MLFLOW_S3_ENDPOINT_URL"] = 'http://localhost:9000'
os.environ["AWS_ENDPOINT_URL_S3"] = 'http://localhost:9001'


mlflow.set_tracking_uri('http://localhost:5005')

experiment_name = "default_model_experiment"

if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(name=experiment_name)

experiment = mlflow.get_experiment_by_name(experiment_name)

# Obtener las variables de entorno (puedes ajustarlas según sea necesario)
pg_user = os.getenv('PG_USER', 'airflow')
pg_password = os.getenv('PG_PASSWORD', 'airflow')
pg_host = 'localhost'  # Si estás ejecutando el script desde dentro del contenedor, usa 'postgres'
pg_port = os.getenv('PG_PORT', '5440')
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

xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, enable_categorical=True)
print(X_train.info())
xgb_regressor.fit(X_train, y_train)

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
    mlflow.sklearn.log_model(xgb_regressor, 'xgb_regressor')



print('Finalizado')