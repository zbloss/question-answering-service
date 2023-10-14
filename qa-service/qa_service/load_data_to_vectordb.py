import json
import os
from typing import Dict, List

import weaviate

from qa_service.models.wine_item import WineItem

WEAVIATE_URL = "http://localhost"
WEAVIATE_PORT = "8123"

WINE_FILEPATH = (
    "C:/Users/altoz/Projects/question-answering-service/data/winemag-data-130k-v2.json"
)


def load_data_to_weaviate(
    weaviate_url: str, weaviate_port: str, wine_items: List[WineItem]
) -> dict:
    """
    Given the array `wine_items`, this function
    will embed the relevant text and upload the
    embeddings and metadata to the Weaviate
    instance available at:
    http://<weaviate_url>:<weaviate_port>

    Arguments:
        weaviate_url : str
            URL or Host that your instance is available
            on.
        weaviate_port : str
            The port your instance is available on.
        wine_items : List[WineItems]
            A python list of WineItems.

    Returns:
        metadata : dict
            Dictionary containing the metadata-response
            received from the Weaviate instance after
            uploading data.
    """

    pass
