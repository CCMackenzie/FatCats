from pydantic import BaseModel, Field, field_validator

from pathlib import Path
from typing import Sequence 

from pydantic import BaseModel, PositiveInt, Field

class FatCatsConfig(BaseModel):
    number_of_players: PositiveInt = Field(..., ge=2, description="Number of players in the game.")
    trick_cards_per_player: PositiveInt = Field(..., description="Initial hand size dealt to each player.")
    allow_multi_bid: bool = Field(
        False,
        description="If True, a player may bet multiple trick cards in one round."
    )
    treat_card_values: Sequence[int] = Field(
        default_factory=lambda: [5, 10, 15, 20],
        description="Possible face values for each randomly-generated treat card",
    )
    treat_deck_size: PositiveInt = Field(
        default=9, description="Number of treat cards revealed per game",
    )

    trick_card_values: Sequence[int] = Field(
        default_factory=lambda: [5, 10, 12, 15],
        description="Population to sample each player's trick hand from",
    )

    @classmethod
    def from_path(cls, path: str | Path) -> "FatCatsConfig":
        """Load & validate a config JSON file in one line."""
        data = Path(path).read_bytes()
        return cls.model_validate_json(data)