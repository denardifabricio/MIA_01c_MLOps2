#!/bin/sh
set -e

# Dirección del dashboard de nuclio
export NUCTL_DASHBOARD_URL="http://nuclio:8070"
PROJECT_NAME="predict-house-price-project"

echo "Creando el proyecto de Nuclio..."
nuctl create project "${PROJECT_NAME}" || true

echo "Desplegando la función de prueba..."

nuctl deploy --path /nuclio-functions/predict_house_price_function

echo "¡Función desplegada con éxito!"