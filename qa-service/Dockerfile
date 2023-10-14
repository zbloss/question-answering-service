FROM python:3.9.13

RUN pip install -U pip

RUN pip install poetry

COPY . /app

WORKDIR /app

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

ENV HOST='0.0.0.0' \
    PORT='8000' \
    WEAVIATE_HOST='localhost' \
    WEAVIATE_PORT='8123' \
    COLLECTION_NAME='WineReview' \
    MODEL_NAME='distilbert-base-uncased' \
    FILEPATH='data/winemag-data-130k-v2.json'

CMD ["python", "api/main.py"]