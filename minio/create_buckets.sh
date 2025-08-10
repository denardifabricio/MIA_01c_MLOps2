#!/bin/sh
sleep 25

/usr/bin/mc alias set s3 "${S3_ENDPOINT_URL}" "${S3_ACCESS_KEY}" "${S3_SECRET_KEY}"
if [ $? -ne 0 ]; then
  echo "Error configurando el host de S3."
  exit 1
fi

if /usr/bin/mc ls s3/${MLFLOW_BUCKET_NAME} >/dev/null 2>&1; then
  echo "Bucket mlflow ya existe."
else
  /usr/bin/mc mb s3/${MLFLOW_BUCKET_NAME} || exit 1
  /usr/bin/mc policy download s3/${MLFLOW_BUCKET_NAME} || exit 1
fi

if /usr/bin/mc ls s3/${DATA_REPO_BUCKET_NAME} >/dev/null 2>&1; then
  echo "Bucket data ya existe."
else
  /usr/bin/mc mb s3/${DATA_REPO_BUCKET_NAME} || exit 1
  /usr/bin/mc policy download s3/${DATA_REPO_BUCKET_NAME} || exit 1
fi

exit 0