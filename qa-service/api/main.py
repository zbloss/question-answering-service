from qa_service.weaviate_layer import WeaviateLayer
from qa_service.utils import read_json_file, validate_wine_objects
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title='NLP Wine Review Query API')

@app.get("/healthcheck")
def read_root():
    return 200

@app.post('/query-text')
def query_text(payload):

    query_results = weaviate_layer.query_collection(
        COLLECTION_NAME, 
        query_text=payload
    )
    return query_results


if __name__ == '__main__':

    HOST = os.getenv('HOST') or '0.0.0.0'
    PORT = os.getenv('PORT') or '8000'
    WEAVIATE_HOST = os.getenv('WEAVIATE_HOST') or 'localhost'
    WEAVIATE_PORT = os.getenv('WEAVIATE_PORT') or '8123'
    COLLECTION_NAME = os.getenv('COLLECTION_NAME') or 'WineReview'
    MODEL_NAME = os.getenv('MODEL_NAME') or 'distilbert-base-uncased'
    FILEPATH = os.getenv('FILEPATH') or 'data/winemag-data-130k-v2.json'

    weaviate_layer = WeaviateLayer(WEAVIATE_HOST, WEAVIATE_PORT, MODEL_NAME)

    print(f'Loading data into Weaviate...')
    wine_objects: list = read_json_file(FILEPATH)
    wine_items = validate_wine_objects(wine_objects[:200])
    weaviate_layer.recreate_collection_class(COLLECTION_NAME)
    weaviate_layer.upload_wine_items(COLLECTION_NAME, wine_items, batch_size=100)
    
    uvicorn.run(app, host=HOST, port=PORT)

