import boto3
from botocore.client import Config
import os
from botocore.exceptions import ClientError
import botocore.session
import io
from google.cloud import storage

# Configuración para Google Cloud Storage usando la API S3 compatible
gcs_endpoint = 'https://storage.googleapis.com'
gcs_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
gcs_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

if not gcs_access_key or not gcs_secret_key:
    raise ValueError('Faltan las variables de entorno AWS_ACCESS_KEY_ID o AWS_SECRET_ACCESS_KEY')

bucket_name = 'tubucketgcs2'  # Debe existir previamente en GCS
gcs_region = 'us-east-1'  # GCS ignora la región, pero boto3 la requiere
object_key = 'scaler/ejemplo.txt'
contenido = 'Hola desde boto3 usando Google Cloud Storage!'

# Crear el cliente S3 apuntando a GCS
s3 = boto3.client(
     's3',
     endpoint_url=gcs_endpoint,
     aws_access_key_id=gcs_access_key,
     aws_secret_access_key=gcs_secret_key,
     region_name=gcs_region,
     config=Config(
        signature_version='s3v4',
        s3={'addressing_style': 'virtual'},
        request_checksum_calculation='when_required',
        response_checksum_validation='when_required'
     ))    


# session = botocore.session.Session()
# s3 = session.create_client(
#     's3',
#     endpoint_url = gcs_endpoint,
#     aws_access_key_id = gcs_access_key,
#     aws_secret_access_key = gcs_secret_key,
#     )




# Verificar que el bucket existe (no intentar crearlo, solo verificar)
def bucket_exists(s3, bucket_name):
    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        print(f'Error al verificar el bucket: {e}')
        return False

# Descargar archivo de ejemplo
response = s3.get_object(Bucket="mlops2-data", Key="train_data.xlsx")
with open("train_data.xlsx", "wb") as f:
    f.write(response["Body"].read())
print("Archivo train_data.xlsx guardado localmente.")


# Crear el bucket si no existe
if not bucket_exists(s3, bucket_name):
    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f'Bucket {bucket_name} creado exitosamente.')
    except ClientError as e:
        print(f'Error al crear el bucket: {e}')
        if hasattr(e, "response"):
            print(f'Response: {e.response}')

# Subir un archivo de ejemplo (solo parámetros mínimos)
try:
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=response["Body"].read())
    print(f'Archivo {object_key} subido exitosamente a {bucket_name} en Google Cloud Storage.')
except ClientError as e:
    print(f'Error al subir el archivo: {e}')
    if hasattr(e, "response"):
        print(f'Response: {e.response}')


# Subir archivo usando upload_fileobj (alternativa a put_object)
try:
    file_obj = io.BytesIO(contenido.encode('utf-8'))

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_key)
    blob.upload_from_string(file_obj)


    print(f'Archivo {object_key} subido exitosamente a {bucket_name} usando upload_fileobj.')
except ClientError as e:
    print(f'Error al subir el archivo con upload_fileobj: {e}')
    if hasattr(e, "response"):
        print(f'Response: {e.response}')


# Listar buckets y objetos para verificar acceso
try:
    print('Buckets disponibles:')
    buckets = s3.list_buckets()
    for b in buckets.get('Buckets', []):
        print(f'- {b["Name"]}')
    print(f'Objetos en el bucket {bucket_name}:')
    objs = s3.list_objects_v2(Bucket=bucket_name)
    for obj in objs.get('Contents', []):
        print(f'- {obj["Key"]}')
except ClientError as e:
    print(f'Error al listar buckets/objetos: {e}')
    if hasattr(e, "response"):
        print(f'Response: {e.response}')