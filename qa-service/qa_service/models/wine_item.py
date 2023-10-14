from typing import Optional

from pydantic import BaseModel


class WineItem(BaseModel):
    """
    A WineItem object contains only the relevant
    fields for our project from the wine dataset.

    It inherits from the Pydantic BaseModel to
    allow for schema validation.
    """

    title: str
    description: str
    variety: str
    region_1: Optional[str] = None
    region_2: Optional[str] = None
    country: str

    @property
    def region(self):
        region = (
            f"{self.region_1} {self.region_2}"
            if self.region_2 is not None
            else self.region_1
        )

        if self.region_1 == self.region_2 == None:
            region = "No available region"

        return region
