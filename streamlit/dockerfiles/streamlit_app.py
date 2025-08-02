import streamlit as st
import requests

# URL del servicio FastAPI
API_URL = "http://fastapi:8800/predict"

# Título de la aplicación
st.title("Predicción de precio de propiedades en Capital Federal")

# Campos para que el usuario ingrese datos
expenses_amount = st.number_input("Monto de Expensas en ARS", min_value=0.0, format="%.2f")
total_mts = st.number_input("Metros Totales", min_value=0.0, format="%.2f")
covered_mts = st.number_input("Metros Cubiertos", min_value=0.0, format="%.2f")
rooms = st.number_input("Cantidad de Habitaciones", min_value=0, step=1)
bedrooms = st.number_input("Cantidad de Dormitorios", min_value=0, step=1)
bathrooms = st.number_input("Cantidad de Baños", min_value=0, step=1)
garages = st.number_input("Cantidad de Cocheras", min_value=0, step=1)
antique = st.number_input("Antigüedad de la Propiedad (en años)", min_value=0, step=1)

# Botón para procesar la información
if st.button("Enviar"):
    # Crea un diccionario con los datos del formulario
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
    
    # Enviar los datos a la API usando requests.post
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            st.success("Datos enviados correctamente")
            prediction = response.json().get("prediction")  
            st.write(f"El valor de la propiedad asciende a U$S: {prediction}") # Muestra la respuesta de FastAPI
        else:
            st.error(f"Error al enviar los datos: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")