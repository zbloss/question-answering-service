from qa_service.weaviate_layer import WeaviateLayer
from qa_service.utils import read_json_file, validate_wine_objects
from fastapi import FastAPI
import uvicorn
import os
import sys
import logging
import requests

app = FastAPI(title="NLP Wine Review Query API")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@app.get("/healthcheck")
def read_root():
    return 200


@app.post("/query-text")
def query_text(payload):

    query_results: list = weaviate_layer.query_collection(
        COLLECTION_NAME, query_text=payload
    )
    if len(query_results) < 1:
        return "I'm sorry but I am unable to find a relevant result."

    top_result: dict = query_results[0]
    top_result['query'] = payload

    response = requests.post(
        f"{CONVERSATIONAL_HOST}:{CONVERSATIONAL_PORT}/nli", 
        json=top_result
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Failed to get NLI: {response.content}')

    return top_result


if __name__ == "__main__":
    HOST = os.getenv("HOST") or "0.0.0.0"
    PORT = os.getenv("PORT") or "8000"
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST") or "http://localhost"
    WEAVIATE_PORT = os.getenv("WEAVIATE_PORT") or "8123"
    CONVERSATIONAL_HOST = os.getenv("CONVERSATIONAL_HOST") or "http://localhost"
    CONVERSATIONAL_PORT = os.getenv("CONVERSATIONAL_PORT") or "8001"
    COLLECTION_NAME = os.getenv("COLLECTION_NAME") or "WineReview"
    MODEL_NAME = os.getenv("MODEL_NAME") or "distilbert-base-uncased"
    FILEPATH = os.getenv("FILEPATH") or "data/winemag-data-130k-v2.json"
    BATCH_SIZE = os.getenv("BATCH_SIZE") or 100
    MODEL_BATCH_SIZE = os.getenv("MODEL_BATCH_SIZE") or 8
    MAX_NUMBER_OF_DOCUMENTS = os.getenv("MAX_NUMBER_OF_DOCUMENTS") or 50

    PORT = int(PORT)
    WEAVIATE_PORT = int(WEAVIATE_PORT)
    BATCH_SIZE = int(BATCH_SIZE)
    MAX_NUMBER_OF_DOCUMENTS = (
        int(MAX_NUMBER_OF_DOCUMENTS)
        if MAX_NUMBER_OF_DOCUMENTS is not None
        else MAX_NUMBER_OF_DOCUMENTS
    )
    MODEL_BATCH_SIZE = (
        int(MODEL_BATCH_SIZE) if MODEL_BATCH_SIZE is not None else MODEL_BATCH_SIZE
    )

    weaviate_layer = WeaviateLayer(WEAVIATE_HOST, WEAVIATE_PORT, MODEL_NAME)

    print(f"Loading data into Weaviate...")
    logging.info(f"Loading data into Weaviate...")
    wine_objects: list = read_json_file(FILEPATH)
    wine_items = validate_wine_objects(wine_objects)
    if MAX_NUMBER_OF_DOCUMENTS is None:
        MAX_NUMBER_OF_DOCUMENTS = len(wine_items)

    wine_items = wine_items[:MAX_NUMBER_OF_DOCUMENTS]
    weaviate_layer.recreate_collection_class(COLLECTION_NAME)

    print(f"WEAVIATE_BATCH_SIZE: {BATCH_SIZE}")
    print(f"MODEL_BATCH_SIZE: {MODEL_BATCH_SIZE}")

    logging.info(f"WEAVIATE_BATCH_SIZE: {BATCH_SIZE}")
    logging.info(f"MODEL_BATCH_SIZE: {MODEL_BATCH_SIZE}")

    weaviate_layer.upload_wine_items(
        collection_name=COLLECTION_NAME,
        wine_items=wine_items,
        batch_size=BATCH_SIZE,
        model_batch_size=MODEL_BATCH_SIZE,
    )

    uvicorn.run(app, host=HOST, port=PORT)
