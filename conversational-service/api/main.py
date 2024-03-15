import logging
import os
import sys

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from transformers import pipeline


app = FastAPI(title="Wine Review Natural Language API Interface")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class ModelRequest(BaseModel):
    query: str
    chunk_index: int
    country: str
    description: str
    region: Optional[str]
    title: str
    variety: str


@app.get("/healthcheck")
def read_root():
    return 200


@app.post("/nli")
def natural_language_interface(data: ModelRequest):
    query = data.query
    country = data.country
    description = data.description
    region = data.region
    title = data.title
    variety = data.variety

    region_information: str = f''
    if region is not None:
        region_information: str = f', specifically from the {region} region'

    context = f'''Please answer the following question as honestly as you can. If you do not
    know the answer please reply with \"I am sorry but I do not know.\"

    Context: The {title} is a {variety} wine from the country of {country}{region_information}.
             Experts describe the wine as \"{description}\"
    '''

    answer = qa_pipeline(
        question=query,
        context=context
    )

    result_with_answer: dict = data.model_dump()
    result_with_answer['answer'] = answer['answer']
    return result_with_answer


if __name__ == "__main__":
    HOST = os.getenv("HOST") or "0.0.0.0"
    PORT = os.getenv("PORT") or "8001"
    MODEL_NAME = os.getenv("MODEL_NAME") or "distilbert-base-uncased-distilled-squad"

    qa_pipeline = pipeline("question-answering", model=MODEL_NAME)

    PORT = int(PORT)
    uvicorn.run(app, host=HOST, port=PORT)
