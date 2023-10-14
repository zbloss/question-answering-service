# https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
FROM python:3.9.13

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip

RUN pip install poetry

COPY . /app

WORKDIR /app

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV QUERY_HOST='http://query_api' \
    QUERY_PORT='8000' \
    QUERY_ENDPOINT='query-text'

ENTRYPOINT ["streamlit", "run", "qa_ui/main.py", "--server.port=8501", "--server.address=0.0.0.0"]