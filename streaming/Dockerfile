
FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY consumer_fastapi_websocket.py /code/consumer_fastapi_websocket.py

CMD ["fastapi", "run", "consumer_fastapi_websocket.py", "--port", "80"]