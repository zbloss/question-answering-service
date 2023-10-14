import json
import tempfile
from typing import List

import pytest

from qa_service.models.wine_item import WineItem
from qa_service.utils import read_json_file, validate_wine_objects


class TestUtils:
    example_wine_object: dict = {
        "points": "87",
        "title": "Nicosia 2013 Vulkà Bianco  (Etna)",
        "description": "Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity.",
        "taster_name": "Kerin O'Keefe",
        "taster_twitter_handle": "@kerinokeefe",
        "price": None,
        "designation": "Vulkà Bianco",
        "variety": "White Blend",
        "region_1": "Etna",
        "region_2": None,
        "province": "Sicily & Sardinia",
        "country": "Italy",
        "winery": "Nicosia",
    }
    temporary_file = tempfile.NamedTemporaryFile(delete=False)
    temporary_file.write(json.dumps(example_wine_object).encode())
    temporary_file.seek(0)
    temporary_filepath: str = temporary_file.name

    def test_read_json_file(self):
        file_contents: dict = read_json_file(self.temporary_filepath)
        assert file_contents == self.example_wine_object

    def test_validate_wine_objects(self):
        total_number_of_objects: int = 3
        wine_objects: list = [
            self.example_wine_object for _ in range(total_number_of_objects)
        ]
        wine_items: List[WineItem] = validate_wine_objects(wine_objects)

        assert isinstance(
            wine_items, list
        ), f"wine_items ({type(wine_items)}) is not a List as expected."
        for wine_item_index, wine_item in enumerate(wine_items):
            failure_message = f"{wine_item_index}. wine_item ({type(wine_item)}) is not a WineItem type as expected."
            assert isinstance(wine_item, WineItem), failure_message
