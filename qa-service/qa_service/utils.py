import json
import os
from typing import Dict, List

from qa_service.models.wine_item import WineItem

WEAVIATE_URL = "http://localhost"
WEAVIATE_PORT = "8123"

WINE_FILEPATH = (
    "C:/Users/altoz/Projects/question-answering-service/data/winemag-data-130k-v2.json"
)


def read_json_file(filepath: str) -> List[Dict[str, str]]:
    """
    Reads the file at `filepath` and loads
    the contents as a JSON or JSON-Lines
    object.

    Arguments:
        filepath : str
            Filepath to read and load as JSON.

    Returns:
        data : List[Dict[str, str]]
            Contents of `filepath` read and
            loaded as a JSON-Lines like object.
    """

    assert os.path.isfile(
        filepath
    ), f"The filepath provided ({filepath}) is not a valid file."

    with open(filepath, "r") as f:
        data = f.read()

    data = json.loads(data)
    return data


def validate_wine_objects(wine_objects: List[Dict[str, str]]) -> List[WineItem]:
    """
    Loops over the dictionaries present in the
    `wine_objects` list. Those are loaded and
    validated via our `models/wine_item.WineItem`
    class and returned as a list of WineItems.

    Arguments:
        wine_objects : List[Dict[str, str]])
            List of dictionaries containing wine
            reviews.

    Returns:
        wine_items : List[WineItem]
            List of WineItem objects with only
            the fields we care about for this
            project extracted.
    """

    wine_items = []
    for wine_object in wine_objects:
        # TODO: handle exceptions where wine items
        #       fail to be created.
        wine_items.append(WineItem(**wine_object))
    return wine_items
