FROM tiangolo/uvicorn-gunicorn:python3.11-2022-11-25

LABEL maintainer="phamdt203@gmail.com"

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /faceRecognition

COPY requirements.txt /faceRecognition/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./