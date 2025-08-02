import logging

from airflow import DAG
from airflow.operators.python_operator import PythonOperator # type: ignore
from airflow.utils.dates import days_ago # type: ignore

import datetime
import numpy as np
import pandas as pd

import boto3
from botocore.client import Config
from io import BytesIO
import pickle

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder



default_args = {
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

dag = DAG(
    'etl',
    default_args=default_args,
    description='Proceso ETL',
    schedule_interval=None,
    start_date=days_ago(2),
    tags=['ETL']
)



def get_data(**kwargs):
    # Configurar conexión a MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url='http://s3:9000',  # URL de tu contenedor MinIO
        aws_access_key_id='minio',  # Cambiar si tienes claves diferentes
        aws_secret_access_key='minio123',  # Cambiar si tienes claves diferentes
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'  # Puedes cambiar la región
    )

    bucket_name = 'data'  # Nombre del bucket
    file_key = 'train_data.xlsx'  # Nombre del archivo en el bucket

    # Descargar el archivo como objeto
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    file_content = response['Body'].read()

    # Usar pandas para leer el archivo Excel
    df = pd.read_excel(BytesIO(file_content), engine='openpyxl')

    # Conectar a PostgreSQL
    engine = create_engine('postgresql+psycopg2://airflow:airflow@postgres/airflow')

    # Guardar el DataFrame en PostgreSQL
    df.to_sql('data', engine, if_exists='replace', index=False)
    df.to_csv("./data.csv", index=False)
    kwargs['ti'].xcom_push(key='data', value=df.to_dict())

    logging.info("Datos guardados en PostgreSQL en la tabla: data")
    logging.info("Datos guardados en csv en la tabla: data")
    logging.info("Datos guardados en XCOM con key: data")

def drop_columns_and_values(**kwargs):
    df = pd.read_csv("./data.csv")

    logging.info("Eliminación por imposibilidad de estimar datos faltantes: 'number_of_floors','apartments_per_floor'")
    df.drop(
        columns=['number_of_floors','apartments_per_floor'],
        inplace=True,
        errors='ignore'
        )
    
    logging.info("Eliminacion por no aportar al análisis: 'url','publisher_id','city_id','state_id','country_id','publisher_name','name','address'")
    df.drop(
        columns=['url','publisher_id','city_id','state_id','country_id','publisher_name','name','address'],
        inplace=True,
        errors='ignore'
        )
    
    logging.info("Nos quedamos unicamente con: 'Apartamento','PH','Casa' ")
    df = df[df['real_estate_type'].isin(['Apartamento','PH','Casa'])]

    logging.info("Eliminacion de valores de operation_amount (target) en 0")
    df = df[df['operation_amount'] != 0]

    df.to_csv("./transformed_data.csv", index=False)
    kwargs['ti'].xcom_push(key='drop_columns_and_values', value=df.to_dict())
    logging.info("Datos guardados en XCOM con key: drop_columns_and_values")
    logging.info("Datos guardados en csv: transformed_data")

def estimate_missing_values(**kwargs):
    df = pd.read_csv("./transformed_data.csv")

    logging.info("Reemplazamos garages faltantes por 0")
    df['garages'] = df['garages'].fillna(0)

    logging.info("Reemplazamos bathrooms faltantes por 1")
    df['bathrooms'] = df['bathrooms'].fillna(1)


    logging.info("""
        En el caso de la variable "bedrooms," seguiremos el siguiente enfoque:

        - Completaremos con 0 aquellos listados donde la variable "rooms" sea igual a 1 (Monoambientes).
        - Para aquellos casos que tengan los valores de "rooms" y "bathrooms" completos, realizaremos la resta entre estas variables para obtener una estimación más aproximada.
        - Para el resto de los casos, dejaremos el valor en 0.
    """)
    df['bedrooms'] = np.where(df['rooms'] == 1, 0, df['bedrooms'])

    condition = df['bedrooms'].isna() & df['rooms'].notna() & df['bathrooms'].notna()
    df.loc[condition, 'bedrooms'] = df.loc[condition, 'rooms'] - df.loc[condition, 'bathrooms']

    logging.info("""
        Para los registros en los que el valor de metros cubiertos sea mayor que el de metros totales, 
        intercambiaremos los valores de estas columnas para corregir el error.
    """)
    condition = df['covered_mts'] > df['total_mts']

    # Intercambiar valores de A y B donde la condición se cumple
    df.loc[condition, ['total_mts', 'covered_mts']] = df.loc[condition, ['covered_mts', 'total_mts']].values

    logging.info("""
        Para los registros que no tengan metros cubiertos, lo completamos con la cantidad de metros totales para tener un valor aproximado
    """)
    condition = df['covered_mts'].isna()
    df.loc[condition, 'covered_mts'] = df.loc[condition, 'total_mts']

    logging.info("""
        Para la orientación, completaremos con "desconocido" aquellos registros sin valor, 
        ya que los valores faltantes para esta variable rondan el 50% y no es un valor inferible; 
        sin embargo, creemos que la orientación puede influir en el valor de un inmueble.
    """)
    df['orientation'].fillna('desconocido', inplace=True)

    logging.info("""
        En el caso de la variable "building_layout," reemplazaremos los valores faltantes con la moda del conjunto de datos.
    """)
    moda_building_layout = df['building_layout'].mode()[0]
    df['building_layout'].fillna(moda_building_layout, inplace=True)

    logging.info("""
        Existen registros que están en moneda ARS ($). Estos los transformaremos a USD utilizando 
        el tipo de cambio (TC) a la fecha de las publicaciones (todos los registros se tomaron el mismo día 
        y se asume que las expensas están actualizadas a dicha fecha) para así unificar los análisis.
    """)
    dolar_bna = 909.5
    dolar_blue = 1220
    currency_rate = (dolar_bna + dolar_blue) / 2
    df['operation_amount'] = np.where(df['operation_currency'] == '$', df['operation_amount'] / currency_rate, df['operation_amount'])
    df['operation_currency'] = 'USD'
    df['expenses_amount'] = np.where(df['expenses_currency'] == 'ARS', df['expenses_amount'] / currency_rate, df['expenses_amount'])
    df['expenses_currency'] = 'USD'

    df.to_csv("./transformed_data.csv", index=False)
    kwargs['ti'].xcom_push(key='estimate_missing_values', value=df.to_dict())
    logging.info("Datos guardados en XCOM con key: estimate_missing_values")
    logging.info("Datos guardados en csv: transformed_data")

def string_transformation(**kwargs):
    df = pd.read_csv("./transformed_data.csv")

    logging.info("""
        La columna "antique" no solo contiene registros de antigüedad en años, sino también valores como 
        "A estrenar" o "En construcción." Para estos últimos dos casos, crearemos variables dummy y 
        reemplazaremos sus valores por 0. Además, para aquellos registros con antigüedades mayores a 1000 años, 
        realizaremos la resta entre 2024 y el valor de la antigüedad para obtener un valor real.
    """)

    # Valores específicos para los cuales queremos crear dummies
    values_to_dummy = ['A estrenar', 'En construcción']

    # Crear una columna temporal que contiene solo los valores específicos y None para los demás
    df['temp'] = df['antique'].apply(lambda x: x if x in values_to_dummy else None)

    # Generar las columnas dummies
    dummies_ant = pd.get_dummies(df['temp'], prefix='antique')

    df = df.join(dummies_ant)
    df.drop(columns=['temp'], inplace=True)

    condition = df['antique'].isin({'A estrenar','En construcción'})
    df.loc[condition, 'antique'] = 0
    df['antique'] = df['antique'].astype('float')

    condition = df['antique'] > 1000
    df.loc[condition, 'antique'] = 2024 - df.loc[condition, 'antique']

    logging.info("""
        Para la variable "building_layout," generaremos variables dummy. Esto permitirá que cada 
        categoría de disposición del edificio sea representada de manera binaria, facilitando así su 
        incorporación en el modelo predictivo.
    """)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_array = encoder.fit_transform(df[['building_layout']].astype(str)).astype(int)

    df = pd.concat([df, pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['building_layout']))], axis=1)

    df = pd.get_dummies(df, columns=['real_estate_type','orientation'], drop_first=True)

    df.to_csv("./transformed_data.csv", index=False)
    kwargs['ti'].xcom_push(key='string_transformation', value=df.to_dict())
    logging.info("Datos guardados en XCOM con key: string_transformation")
    logging.info("Datos guardados en csv: transformed_data")


def outlier_elimination(**kwargs):
    df = pd.read_csv("./transformed_data.csv")

    logging.info("""
        Eliminaremos aquellos outliers que distorsionan el conjunto de datos. Para ello, aplicaremos 
        como regla general la eliminación de los registros que se encuentren por encima del percentil 0.95.
    """)
    # Lista de columnas que queremos filtrar
    columns_to_filter = ['operation_amount', 'expenses_amount', 'total_mts', 'covered_mts', 'rooms', 'bedrooms', 'bathrooms', 'garages', 'antique']

    # Calcular el percentil para cada columna en la lista
    percentile = df[columns_to_filter].quantile(0.95)

    # Filtrar el DataFrame para eliminar valores por encima del percentil en las columnas seleccionadas
    for col in columns_to_filter:
        df = df[df[col] <= percentile[col]]
    
    logging.info("""
        También nos quedaremos con aquellos valores de "operation_amount" que sean mayores de $35.000,00.
    """)
    df = df[df['operation_amount'] > 35000]

    df = df[['expenses_amount','total_mts','covered_mts','rooms','bedrooms','bathrooms','garages','antique','operation_amount']]

    df.to_csv("./transformed_data.csv", index=False)
    kwargs['ti'].xcom_push(key='outlier_elimination', value=df.to_dict())
    logging.info("Datos guardados en XCOM con key: outlier_elimination")
    logging.info("Datos guardados en csv: transformed_data")

    
def split_data(**kwargs):
    df = pd.read_csv("./transformed_data.csv")

    df[df.select_dtypes(['bool']).columns] = df.select_dtypes(['bool']).astype(int)

    X = df.drop(columns=['operation_amount'])
    y = df['operation_amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Conectar a PostgreSQL
    engine = create_engine('postgresql+psycopg2://airflow:airflow@postgres/airflow')

    logging.info("X_train guardado en PostgreSQL en la tabla: X_train")
    logging.info("X_train en XCOM con key: X_train")
    kwargs['ti'].xcom_push(key='X_train', value=X_train.to_dict())
    X_train.to_sql('X_train', engine, if_exists='replace', index=False)

    logging.info("X_test guardado en PostgreSQL en la tabla: X_test")
    logging.info("X_test en XCOM con key: X_test")
    kwargs['ti'].xcom_push(key='X_test', value=X_test.to_dict())
    X_test.to_sql('X_test', engine, if_exists='replace', index=False)

    logging.info("y_train guardado en PostgreSQL en la tabla: y_train")
    logging.info("y_train en XCOM con key: y_train")
    kwargs['ti'].xcom_push(key='y_train', value=y_train.to_dict())
    y_train.to_sql('y_train', engine, if_exists='replace', index=False)

    logging.info("y_test guardado en PostgreSQL en la tabla: y_test")
    logging.info("y_test en XCOM con key: y_test")
    kwargs['ti'].xcom_push(key='y_test', value=y_test.to_dict())
    y_test.to_sql('y_test', engine, if_exists='replace', index=False)

    X_train.to_csv("./X_train.csv", index=False)
    X_test.to_csv("./X_test.csv", index=False)
    y_train.to_csv("./y_train.csv", index=False)
    y_test.to_csv("./y_test.csv", index=False)

def scale_data(**kwargs):
    logging.info("Leyendo Datos")
    X_train = pd.read_csv("./X_train.csv")
    X_test = pd.read_csv("./X_test.csv")
    y_train = pd.read_csv("./y_train.csv")
    y_test = pd.read_csv("./y_test.csv")


    logging.info("Datos Leidos")

    # Crear una instancia de StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Ajustar el scaler al conjunto de entrenamiento y transformar
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    # Serializar el objeto StandardScaler
    scaler_data_x = pickle.dumps(scaler_X)
    scaler_data_y = pickle.dumps(scaler_y)

    # Subir a S3
    s3 = boto3.client('s3')
    bucket_name = 'data'
    scaler_filename = '/scalers/scaler_X.pkl'
    s3.put_object(Bucket=bucket_name, Key=scaler_filename, Body=scaler_data_x)

    scaler_filename = '/scalers/scaler_y.pkl'
    s3.put_object(Bucket=bucket_name, Key=scaler_filename, Body=scaler_data_y)

    # Transformar el conjunto de testeo
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    y_train_scaled = pd.DataFrame(y_train_scaled, columns=y_train.columns)
    y_test_scaled = pd.DataFrame(y_test_scaled, columns=y_test.columns)

    # Conectar a PostgreSQL
    engine = create_engine('postgresql+psycopg2://airflow:airflow@postgres/airflow')

    logging.info("X_train_scaled guardado en PostgreSQL en la tabla: X_train_scaled")
    logging.info("X_train_scaled en XCOM con key: X_train_scaled")
    kwargs['ti'].xcom_push(key='X_train_scaled', value=X_train_scaled.to_dict())
    X_train_scaled.to_sql('X_train_scaled', engine, if_exists='replace', index=False)

    logging.info("X_test_scaled guardado en PostgreSQL en la tabla: X_test_scaled")
    logging.info("X_test_scaled en XCOM con key: X_test_scaled")
    kwargs['ti'].xcom_push(key='X_test_scaled', value=X_test_scaled.to_dict())
    X_test_scaled.to_sql('X_test_scaled', engine, if_exists='replace', index=False)

    logging.info("y_train_scaled guardado en PostgreSQL en la tabla: y_train_scaled")
    logging.info("y_train_scaled en XCOM con key: y_train_scaled")
    kwargs['ti'].xcom_push(key='y_train_scaled', value=y_train_scaled.to_dict())
    y_train_scaled.to_sql('y_train_scaled', engine, if_exists='replace', index=False)

    logging.info("y_test_scaled guardado en PostgreSQL en la tabla: y_test_scaled")
    logging.info("y_test_scaled en XCOM con key: y_test_scaled")
    kwargs['ti'].xcom_push(key='y_test_scaled', value=y_test_scaled.to_dict())
    y_test_scaled.to_sql('y_test_scaled', engine, if_exists='replace', index=False)

    X_train_scaled.to_csv("./X_train_scaled.csv", index=False)
    X_test_scaled.to_csv("./X_test_scaled.csv", index=False)
    y_train_scaled.to_csv("./y_train_scaled.csv", index=False)
    y_test_scaled.to_csv("./y_test_scaled.csv", index=False)



get_data_operator = PythonOperator(
    task_id='get_data',
    python_callable=get_data,
    dag=dag
)

drop_columns_and_values_operator = PythonOperator(
    task_id='drop_columns_and_values',
    python_callable=drop_columns_and_values,
    dag=dag
)

estimate_missing_values_operator = PythonOperator(
    task_id='estimate_missing_values',
    python_callable=estimate_missing_values,
    dag=dag
)

string_transformation_operator = PythonOperator(
    task_id='string_transformation',
    python_callable=string_transformation,
    dag=dag
)

outlier_elimination_operator = PythonOperator(
    task_id='outlier_elimination',
    python_callable=outlier_elimination,
    dag=dag
)

split_data_operator = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    dag=dag
)

scale_data_operator = PythonOperator(
    task_id='scale_data',
    python_callable=scale_data,
    dag=dag
)



get_data_operator >> drop_columns_and_values_operator >> estimate_missing_values_operator >> string_transformation_operator >> outlier_elimination_operator >> split_data_operator >> scale_data_operator
