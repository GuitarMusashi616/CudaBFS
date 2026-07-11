# pyright: basic

from __future__ import annotations
import json
from Effect import Effect

class Item:
    def __init__(self, name: str, cost: int, always_add_effect: Effect, switch_effects: list[Effect]):
        self.name = name
        self.cost = cost
        self.always_add_effect = always_add_effect
        self.switch_effects = switch_effects

    @staticmethod
    def from_json_file(filename: str = 'data/items.json'):
        obj = json.load(open(filename))
        return [Item.from_json(x) for x in obj]
    
    @staticmethod
    def from_json(obj: dict[str, str | dict[str, list[str]]]) -> Item:
        name = obj['Name']
        cost = obj['Cost']
        always_add_effect = Effect.from_str(obj['AlwaysAddEffect']) # type: ignore
        switch_effects = {}
        for item, val in obj['SwitchEffects'].items(): # type: ignore
            item = Effect.from_str(item)
            val = [Effect.from_str(x) for x in val]
            switch_effects[item] = val

        return Item(name, cost, always_add_effect, switch_effects) # type: ignore
    
    def __repr__(self) -> str:
        return f"{self.name}\n{self.always_add_effect}\n{self.switch_effects}\n"