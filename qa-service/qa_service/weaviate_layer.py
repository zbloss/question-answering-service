import time
from typing import List, Tuple, Dict

import requests
import torch
import weaviate

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

    def _embed_model_batch(
        self,
        wine_items: List[WineItem],
        model_batch_size: int,
        start_index: int,
        field_to_embed: str,
        mean_pooling: bool = True,
        return_cls_embedding: bool = False,
    ) -> Tuple[List[Dict[str, str]], List]:
        """
        Generates a batch of WineItems of size `model_batch_size`,
        beginning at `start_index` index of the `wine_items`.

        Arguments:
            wine_items : List[WineItem]
                List of WineItem objects containing a `field_to_embed`
                to be embedded in batches.
            model_batch_size : int
                How many `field_to_embed`s of WineItems to embed at once.
            start_index : int
                The index to start generating the batch from.
                Data in the batch will begin at
                `wine_items[start_index]` and ends at
                `wine_items[start_index + batch_size]`.
            field_to_embed : str
                A key in each individual `WineItem` to be embedded.
            mean_pooling : bool
                True if you wish to apply mean pooling on the embedded
                matrix.
            return_cls_embedding : bool
                True if you wish to return the CLS layer from the
                embedded matrix.
        Returns:
            data_object, vectors : Tuple[List[Dict[str, str]], List]
                A tuple of the data object ready to load to
                a collection as well as the paring vector.
                These lists are matched on index so easy uploading
                to Weaviate.
        """

        end_index: int = start_index + model_batch_size
        end_index: int = len(wine_items) if len(wine_items) < end_index else end_index

        batch: list = wine_items[start_index:end_index]
        
        text_field: list = [batch_.model_dump()[field_to_embed] for batch_ in batch]

        vectors: torch.Tensor = self.embedding_layer(
            text_field, 'max_length', mean_pooling, return_cls_embedding
        ).tolist()

        assert len(vectors) == len(batch) == len(text_field)

        data_objects: list = []
        for index_, wine_item in enumerate(batch):
            data_object = wine_item.model_dump()
            data_object["chunk_index"] = index_
            data_objects.append(data_object)

        return data_objects, vectors

    def upload_wine_items(
        self,
        collection_name: str,
        wine_items: List[WineItem],
        batch_size: int,
        model_batch_size: int = None,
        field_to_embed: str = "description",
        mean_pooling: bool = True,
        return_cls_embedding: bool = False,
    ):
        """
        Uploads a set of data to a given weaviate collection.
        """

        self.weaviate_client.batch.configure(batch_size=batch_size)
        with self.weaviate_client.batch as batch:
            for wine_item_index in range(0, len(wine_items), model_batch_size):
                data_objects, vectors = self._embed_model_batch(
                    wine_items,
                    model_batch_size,
                    wine_item_index,
                    field_to_embed,
                    mean_pooling,
                    return_cls_embedding,
                )

                assert len(data_objects) == len(
                    vectors
                ), f"data_objects ({len(data_objects)}) & vectors ({len(vectors)}) are not the same length."

                for (data_object, vector) in zip(data_objects, vectors):

                    try:
                        batch.add_data_object(
                            data_object,
                            collection_name,
                            vector=vector,
                        )
                    except weaviate.exceptions.UnexpectedStatusCodeException as e:
                        print(
                            f"Unable to load data_object into Vector DB | data_object: {data_object} | Exception: {e}"
                        )
                        pass
                
                print(f'Total Documents Indexed: {wine_item_index + model_batch_size}')

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
