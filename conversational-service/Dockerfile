FROM python:3.9.13

RUN pip install -U pip

RUN pip install poetry

COPY . /conversational

WORKDIR /conversational

RUN poetry config virtualenvs.create false && \
    poetry install

ENV HOST='0.0.0.0' \
    PORT='8001' \
    MODEL_NAME='distilbert-base-uncased'

CMD ["python", "api/main.py"]