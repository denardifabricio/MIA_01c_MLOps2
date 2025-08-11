import pandas as pd
import streamlit as st
import requests
import grpc
import predict_pb2
import predict_pb2_grpc
import logging
import sys


# URL del servicio FastAPI
API_URL = "http://fastapi:8800/predict"
# URL del servicio gRPC
GRPC_URL = "grpc:50051"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("streamlit_app.log")
    ]
)
logger = logging.getLogger(__name__)

# Título de la aplicación
st.title("Predicción de precio de propiedades en Capital Federal (API y gRPC)")

# Campos para que el usuario ingrese datos
expenses_amount = st.number_input("Monto de Expensas en ARS", min_value=0.0, format="%.2f")
total_mts = st.number_input("Metros Totales", min_value=0.0, format="%.2f")
covered_mts = st.number_input("Metros Cubiertos", min_value=0.0, format="%.2f")
rooms = st.number_input("Cantidad de Habitaciones", min_value=0, step=1)
bedrooms = st.number_input("Cantidad de Dormitorios", min_value=0, step=1)
bathrooms = st.number_input("Cantidad de Baños", min_value=0, step=1)
garages = st.number_input("Cantidad de Cocheras", min_value=0, step=1)
antique = st.number_input("Antigüedad de la Propiedad (en años)", min_value=0, step=1)

if st.button("Enviar"):
    data = {
        "expenses_amount": expenses_amount,
        "total_mts": total_mts,
        "covered_mts": covered_mts,
        "rooms": rooms,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "garages": garages,
        "antique": antique
    }

    resultados = []  # Lista para almacenar resultados de cada modelo

    # --- FASTAPI REST ---
    try:
        logger.info(f"Enviando datos a FastAPI: {data}")
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            st.success("REST: Datos enviados correctamente")
            prediction = response.json().get("prediction")
            execution_time = response.json().get("execution_time")
            st.write(f"REST: El valor de la propiedad asciende a U$S: {prediction}")
            if execution_time is not None:
                st.write(f"REST: Tiempo de ejecución: {execution_time:.4f} segundos")
            logger.info(f"Respuesta FastAPI: {prediction}")
            resultados.append({
                "Modelo": "REST",
                "Resultado": round(prediction, 2) if prediction is not None else None,
                "Tiempo de ejecución (s)": round(execution_time, 2) if execution_time is not None else None
            })
        else:
            st.error(f"REST: Error al enviar los datos: {response.status_code}")
            logger.error(f"REST: Error al enviar los datos: {response.status_code}")
    except Exception as e:
        st.error(f"REST: Error: {e}")
        logger.error(f"REST: Error: {e}")

    # --- gRPC ---
    try:
        logger.info(f"Enviando datos a gRPC: {data}")
        channel = grpc.insecure_channel(GRPC_URL)
        stub = predict_pb2_grpc.PropertyPricePredictorStub(channel)
        grpc_request = predict_pb2.PredictRequest(**data)
        grpc_response = stub.Predict(grpc_request)
        st.success("gRPC: Datos enviados correctamente")
        st.write(f"gRPC: El valor de la propiedad asciende a U$S: {grpc_response.prediction}")
        exec_time = getattr(grpc_response, 'execution_time', None)
        if exec_time is not None:
            st.write(f"gRPC: Tiempo de ejecución: {exec_time:.4f} segundos")
        logger.info(f"Respuesta gRPC: {grpc_response.prediction}")
        resultados.append({
            "Modelo": "gRPC",
            "Resultado": round(grpc_response.prediction, 2) if grpc_response.prediction is not None else None,
            "Tiempo de ejecución (s)": round(exec_time, 2) if exec_time is not None else None
        })
    except Exception as e:
        st.error(f"gRPC: Error: {e}")
        logger.error(f"gRPC: Error: {e}")

    # --- GraphQL ---
    try:
        data_gql = {
            "expensesAmount": expenses_amount,
            "totalMts": total_mts,
            "coveredMts": covered_mts,
            "rooms": rooms,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "garages": garages,
            "antique": antique
        }
        logger.info(f"Enviando datos a GraphQL: {data_gql}")
        graphql_url = "http://graphql:8000/graphql"
        query = """
        mutation Predict($inputData: PredictInput!) {
            predict(inputData: $inputData) {
                prediction
                executionTime
            }
        }
        """
        variables = {"inputData": data_gql}
        headers = {"Content-Type": "application/json"}
        graphql_response = requests.post(
            graphql_url,
            json={"query": query, "variables": variables},
            headers=headers
        )
        if graphql_response.status_code == 200:
            result = graphql_response.json()
            prediction = result.get("data", {}).get("predict", {}).get("prediction")
            execution_time = result.get("data", {}).get("predict", {}).get("executionTime")
            if prediction is not None:
                st.success("GraphQL: Datos enviados correctamente")
                st.write(f"GraphQL: El valor de la propiedad asciende a U$S: {prediction}")
                if execution_time is not None:
                    st.write(f"GraphQL: Tiempo de ejecución: {execution_time:.4f} segundos")
                logger.info(f"Respuesta GraphQL: {prediction}")
                resultados.append({
                    "Modelo": "GraphQL",
                    "Resultado": round(prediction, 2) if prediction is not None else None,
                    "Tiempo de ejecución (s)": round(execution_time, 2) if execution_time is not None else None
                })
            else:
                st.error("GraphQL: No se recibió una predicción válida")
                logger.error(f"GraphQL: Respuesta inesperada: {result}")
        else:
            st.error(f"GraphQL: Error al enviar los datos: {graphql_response.status_code}")
            logger.error(f"GraphQL: Error al enviar los datos: {graphql_response.status_code}")
    except Exception as e:
        st.error(f"GraphQL: Error: {e}")
        logger.error(f"GraphQL: Error: {e}")

    if resultados:
        st.markdown("### Comparación de modelos")
        df_resultados = pd.DataFrame(resultados)
        # Formatear las columnas numéricas a 2 decimales
        for col in ["Resultado"]:
            if col in df_resultados.columns:
                df_resultados[col] = df_resultados[col].map(lambda x: f"{x:.2f}" if x is not None and x != "" else "")
        st.table(df_resultados)