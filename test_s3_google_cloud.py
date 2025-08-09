import boto3
from botocore.client import Config
import os
from botocore.exceptions import ClientError

# Configuración para Google Cloud Storage usando la API S3 compatible
gcs_endpoint = 'https://storage.googleapis.com'
gcs_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
gcs_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

if not gcs_access_key or not gcs_secret_key:
    raise ValueError('Faltan las variables de entorno AWS_ACCESS_KEY_ID o AWS_SECRET_ACCESS_KEY')

bucket_name = 'tu-bucket-gcs'  # Debe existir previamente en GCS
gcs_region = 'auto'  # GCS ignora la región, pero boto3 la requiere
object_key = 'ejemplo.txt'
contenido = 'Hola desde boto3 usando Google Cloud Storage!'

# Crear el cliente S3 apuntando a GCS
s3 = boto3.client(
    's3',
    endpoint_url=gcs_endpoint,
    aws_access_key_id=gcs_access_key,
    aws_secret_access_key=gcs_secret_key,
    config=Config(signature_version='s3v4'),
    region_name=gcs_region
)

# Verificar que el bucket existe
def bucket_exists(s3, bucket_name):
    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        print(f'Error al verificar el bucket: {e}')
        return False

if not bucket_exists(s3, bucket_name):
    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f'Bucket {bucket_name} creado exitosamente en Google Cloud Storage.')
    except ClientError as e:
        print(f'Error al crear el bucket: {e}')
        raise RuntimeError(f'El bucket {bucket_name} no existe o no es accesible en GCS.\n\nSolución: Crea el bucket manualmente en Google Cloud Console y asegúrate de que el proyecto tenga una cuenta de facturación activa.')

# Subir un archivo de ejemplo
try:
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=contenido.encode('utf-8'))
    print(f'Archivo {object_key} subido exitosamente a {bucket_name} en Google Cloud Storage.')
except ClientError as e:
    print(f'Error al subir el archivo: {e}')
    raise