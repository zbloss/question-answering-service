import pytest
from pydantic import ValidationError

from qa_service.models.wine_item import WineItem


class TestWineItem:
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

    # No description in the below object making
    # it invalid.
    example_invalid_wine_object: dict = {
        "points": "87",
        "title": "Nicosia 2013 Vulkà Bianco  (Etna)",
        "taster_name": "Kerin O’Keefe",
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

    example_wine_object_with_multiple_regions: dict = {
        "points": "87",
        "title": "Nicosia 2013 Vulkà Bianco  (Etna)",
        "description": "Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity.",
        "taster_name": "Kerin O’Keefe",
        "taster_twitter_handle": "@kerinokeefe",
        "price": None,
        "designation": "Vulkà Bianco",
        "variety": "White Blend",
        "region_1": "Etna",
        "region_2": "Florence",
        "province": "Sicily & Sardinia",
        "country": "Italy",
        "winery": "Nicosia",
    }

    def test_wine_item(self):
        WineItem(**self.example_wine_object)
        with pytest.raises(ValidationError):
            WineItem(**self.example_invalid_wine_object)

    def test_wine_region(self):
        wine_item = WineItem(**self.example_wine_object)
        assert isinstance(
            wine_item.region, str
        ), f"wine_item.region ({wine_item.region}) is not a string as expected."

        assert wine_item.region == self.example_wine_object["region_1"]

    def test_wine_multiple_regions(self):
        wine_item = WineItem(**self.example_wine_object_with_multiple_regions)
        region_1: str = self.example_wine_object_with_multiple_regions["region_1"]
        region_2: str = self.example_wine_object_with_multiple_regions["region_2"]
        expected_region = f"{region_1} {region_2}"

        assert (
            wine_item.region == expected_region
        ), f"wine_item.region ({wine_item.region}) was not concatenated as expected."
