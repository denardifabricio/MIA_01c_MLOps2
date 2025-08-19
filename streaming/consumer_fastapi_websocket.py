import asyncio
from dataclasses import Field, dataclass
import json
from time import sleep
import uuid
from aiokafka import AIOKafkaConsumer
import anyio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from kafka import KafkaConsumer
import requests
from loguru import logger

API_URL = "http://nuclio-nuclio-predict-house-price:8080"
TOPIC_NAME = "appartment_data"
KAFKA_SERVER_URL = "kafka:9092"
KAFKA_GROUP_ID = "fastapi-streaming-group"


def _deserializer(message):
    try:
        return json.loads(message.decode("utf-8"))
    except json.JSONDecodeError:
        if isinstance(message, bytes):
            return message.decode("utf-8")
        return message


async def _consume():
    logger.debug("Starting to consume messages from Kafka...")
    consumer = AIOKafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=KAFKA_SERVER_URL,
        group_id=KAFKA_GROUP_ID,
        value_deserializer=_deserializer,
    )
    logger.debug("Connecting to Kafka...")
    await consumer.start()
    logger.debug("Kafka consumer started, waiting for messages...")
    try:
        async for msg in consumer:
            data = msg.value
            logger.debug(f"Received from Kafka: {data}")
            try:
                response = await anyio.to_thread.run_sync(lambda: requests.post(API_URL, json=data))
                response.raise_for_status()
                logger.debug(f"Response from API: {response.status_code}")
                logger.debug(f"Response content: {response.content}")
                if response.status_code == 200:
                    prediction = response.json().get("prediction")
                    print(f"Prediction from API: {prediction}")
                    yield f'{prediction}\n'
            except Exception as e:
                logger.error(f"Error llamar API: {e}")
                yield f"data: Error al llamar a la API: {e}\n\n"
    except StopIteration:
        print("No hay más mensajes en el topic.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        await consumer.stop()
        print("Kafka consumer stopped.")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Server Side Event Streaming Endpoint
# https://medium.com/@nandagopal05/server-sent-events-with-python-fastapi-f1960e0c8e4b
@app.get("/stream")
async def stream():
    logger.debug("Starting to stream predictions...")
    return StreamingResponse(_consume(), media_type="text/event-stream")


@app.get("/")
def read_root():
    return "Todo ok. Ejecuta /stream para recibir predicciones en tiempo real."
