#!/bin/sh

# Esperar 20 segundos
sleep 20

# Configurar alias de MinIO Client (mc)
mc alias set s3 "${S3_ENDPOINT_URL}" "${S3_ACCESS_KEY}" "${S3_SECRET_KEY}"
if [ $? -ne 0 ]; then
    echo "Error configurando el alias de MinIO."
    exit 1
fi

# Subir el archivo al bucket
mc cp /data/train_data.xlsx "s3/${DATA_REPO_BUCKET_NAME}"
if [ $? -ne 0 ]; then
    echo "Error subiendo el archivo al bucket."
    exit 1
fi