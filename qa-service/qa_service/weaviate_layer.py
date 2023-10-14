import time
from typing import List

import requests
import torch
import weaviate
from tqdm import tqdm

from qa_service.embedding_layer import EmbeddingLayer
from qa_service.models.wine_item import WineItem


class WeaviateLayer:
    def __init__(self, weaviate_url: str, weaviate_port: str, model_name: str) -> None:
        self.weaviate_url: str = weaviate_url
        self.weaviate_port: str = weaviate_port
        self.model_name: str = model_name
        self.embedding_layer: EmbeddingLayer = EmbeddingLayer(model_name, use_gpu=False)

        self.weaviate_client = weaviate.Client(f"http://{weaviate_url}:{weaviate_port}")

    def recreate_collection_class(
        self,
        collection_name: str,
        text_fields_to_upload: List[str] = [
            "title",
            "description",
            "variety",
            "region",
            "country",
        ],
    ):
        """
        Destroys and recreates a weaviate collection.
        Not to be used in production, but handy for this
        assignment.
        """
        if self.weaviate_client.schema.exists(collection_name):
            self.weaviate_client.schema.delete_class(collection_name)

        collection_class: dict = {"class": collection_name, "vectorizer": "none"}
        properties: list = []
        for text_field in text_fields_to_upload:
            properties.append({"name": text_field, "dataType": ["text"]})

        collection_class["properties"] = properties

        self.weaviate_client.schema.create_class(collection_class)

    def upload_wine_items(
        self,
        collection_name: str,
        wine_items: List[WineItem],
        batch_size: int,
        field_to_embed: str = "description",
    ):
        """
        Uploads a set of data to a given weaviate collection.
        """

        self.weaviate_client.batch.configure(batch_size=batch_size)
        with self.weaviate_client.batch as batch:
            for wine_item_index, wine_item in enumerate(
                tqdm(
                    wine_items,
                    desc="Uploading text to Vector DB",
                    total=len(wine_items),
                )
            ):
                data_object = wine_item.model_dump()
                data_object["chunk_index"] = wine_item_index

                wine_review_embedding: torch.Tensor = self.embedding_layer(
                    data_object[field_to_embed]
                )
                wine_review_vector: list = wine_review_embedding.int().tolist()
                if len(wine_review_vector) == 1:
                    wine_review_vector: list = wine_review_vector[0]
                    assert len(wine_review_vector) != 1

                try:
                    batch.add_data_object(
                        data_object,
                        collection_name,
                        vector=wine_review_vector,
                    )
                except weaviate.exceptions.UnexpectedStatusCodeException as e:
                    print(
                        f"Unable to load data_object into Vector DB | data_object: {data_object} | Exception: {e}"
                    )
                    pass

    def query_collection(
        self,
        collection_name: str,
        query_text: str,
        additional_fields_to_return: list = [
            "title",
            "description",
            "variety",
            "region",
            "country",
            "chunk_index",
        ],
        vector_certainty: float = 0.75,
        max_results: int = 3,
    ):
        query_embedding: torch.Tensor = self.embedding_layer(query_text)
        query_vector: list = query_embedding.tolist()
        if len(query_vector) == 1:
            query_vector: list = query_vector[0]
            assert len(query_vector) != 1

        response = (
            self.weaviate_client.query.get(
                class_name=collection_name,
                properties=additional_fields_to_return,
            )
            .with_near_vector({"vector": query_vector, "certainty": vector_certainty})
            .with_limit(max_results)
            .do()
        )
        out = response["data"]["Get"][collection_name]
        return out

    def weaviate_healthcheck(self):
        """Checks to see if the weaviate service is running."""

        # TODO: utilize requests.Sessions for better back-off
        #       and retry logic.

        healthcheck_endpoint = "v1/.well-known/live"
        url = f"http://{self.weaviate_url}:{self.weaviate_port}/{healthcheck_endpoint}"
        response = requests.get(url)

        attempts: int = 0
        while response.status_code != 200 or attempts > 5:
            time.sleep(3)
            response = requests.get(url)
            attempts += 1
        if response.status_code != 200:
            return False
        else:
            return True


if __name__ == "__main__":
    from qa_service.utils import read_json_file, validate_wine_objects

    WEAVIATE_URL = "http://localhost"
    WEAVIATE_PORT = "8123"
    WINE_FILEPATH = "C:/Users/altoz/Projects/question-answering-service/data/winemag-data-130k-v2.json"
    DEBUG = True
    COLLECTION_NAME = "WineReview"

    model_name: str = "distilbert-base-uncased"

    wine_objects: list = read_json_file(WINE_FILEPATH)
    if DEBUG:
        wine_objects = wine_objects[:100]

    wine_items: List[WineItem] = validate_wine_objects(wine_objects)
    weaviate_layer = WeaviateLayer(WEAVIATE_URL, WEAVIATE_PORT, model_name)

    weaviate_layer.recreate_collection_class(COLLECTION_NAME)
    weaviate_layer.upload_wine_items(COLLECTION_NAME, wine_items, batch_size=5)

    is_weaviate_up = weaviate_layer.weaviate_healthcheck()

    query_results = weaviate_layer.query_collection(
        COLLECTION_NAME, query_text="funky and fruity"
    )
    print(query_results)
