#!/bin/sh
sleep 25

/usr/bin/mc alias set s3 http://s3:9000 ${MINIO_ACCESS_KEY:-minio} ${MINIO_SECRET_ACCESS_KEY:-minio123}
if [ $? -ne 0 ]; then
  echo "Error configurando el host de MinIO."
  exit 1
fi

if /usr/bin/mc ls s3/${MLFLOW_BUCKET_NAME:-mlflow} >/dev/null 2>&1; then
  echo "Bucket mlflow ya existe."
else
  /usr/bin/mc mb s3/${MLFLOW_BUCKET_NAME:-mlflow} || exit 1
  /usr/bin/mc policy download s3/${MLFLOW_BUCKET_NAME:-mlflow} || exit 1
fi

if /usr/bin/mc ls s3/${DATA_REPO_BUCKET_NAME:-data} >/dev/null 2>&1; then
  echo "Bucket data ya existe."
else
  /usr/bin/mc mb s3/${DATA_REPO_BUCKET_NAME:-data} || exit 1
  /usr/bin/mc policy download s3/${DATA_REPO_BUCKET_NAME:-data} || exit 1
fi

exit 0