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
# ...existing code...

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

    # --- FASTAPI REST ---
    try:
        logger.info(f"Enviando datos a FastAPI: {data}")
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            st.success("REST: Datos enviados correctamente")
            prediction = response.json().get("prediction")
            st.write(f"REST: El valor de la propiedad asciende a U$S: {prediction}")
            logger.info(f"Respuesta FastAPI: {prediction}")
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
        logger.info(f"Respuesta gRPC: {grpc_response.prediction}")
    except Exception as e:
        st.error(f"gRPC: Error: {e}")
        logger.error(f"gRPC: Error: {e}")

    # --- GraphQL ---
    try:

        data = {
            "expensesAmount": expenses_amount,
            "totalMts": total_mts,
            "coveredMts": covered_mts,
            "rooms": rooms,     
            "bedrooms": bedrooms,
            "bathrooms": bathrooms, 
            "garages": garages,
            "antique": antique
        }
        
        logger.info(f"Enviando datos a GraphQL: {data}")
        graphql_url = "http://graphql:8000/graphql"
        query = """
        mutation Predict($inputData: PredictInput!) {
            predict(inputData: $inputData) {
                prediction
            }
        }
        """
        variables = {"inputData": data}
        headers = {"Content-Type": "application/json"}
        graphql_response = requests.post(
            graphql_url,
            json={"query": query, "variables": variables},
            headers=headers
        )
        if graphql_response.status_code == 200:
            result = graphql_response.json()
            prediction = result.get("data", {}).get("predict", {}).get("prediction")
            if prediction is not None:
                st.success("GraphQL: Datos enviados correctamente")
                st.write(f"GraphQL: El valor de la propiedad asciende a U$S: {prediction}")
                logger.info(f"Respuesta GraphQL: {prediction}")
            else:
                st.error("GraphQL: No se recibió una predicción válida")
                logger.error(f"GraphQL: Respuesta inesperada: {result}")
        else:
            st.error(f"GraphQL: Error al enviar los datos: {graphql_response.status_code}")
            logger.error(f"GraphQL: Error al enviar los datos: {graphql_response.status_code}")
    except Exception as e:
        st.error(f"GraphQL: Error: {e}")
        logger.error(f"GraphQL: Error: {e}")