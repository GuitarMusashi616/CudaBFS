# pyright: basic

from __future__ import annotations
from enum import Enum

class Effect(Enum):
    ANTI_GRAVITY =      (1, 0.54)
    ATHLETIC =          (2, 0.32)
    BALDING =           (3, 0.3)
    BRIGHT_EYED =       (4, 0.4)
    CALMING =           (5, 0.1)
    CALORIE_DENSE =     (6, 0.28)
    CYCLOPEAN =         (7, 0.56)
    DISORIENTING =      (8, 0)
    ELECTRIFYING =      (9, 0.5)
    ENERGIZING =        (10, 0.22)
    EUPHORIC =          (11, 0.18)
    EXPLOSIVE =         (12, 0)
    FOCUSED =           (13, 0.16)
    FOGGY =             (14, 0.36)
    GINGERITIS =        (15, 0.2)
    GLOWING =           (16, 0.48)
    JENNERISING =       (17, 0.42)
    LAXATIVE =          (18, 0)
    LETHAL =            (19, 0)
    LONG_FACED =        (20, 0.52)
    MUNCHIES =          (21, 0.12)
    PARANOIA =          (22, 0)
    REFRESHING =        (23, 0.14)
    SCHIZOPHRENIC =     (24, 0)
    SEDATING =          (25, 0.26)
    SEIZURE_INDUCING =  (26, 0)
    SHRINKING =         (27, 0.6)
    SLIPPERY =          (28, 0.34)
    SMELLY =            (29, 0)
    SNEAKY =            (30, 0.24)
    SPICY =             (31, 0.38)
    THOUGHT_PROVOKING = (32, 0.44)
    TOXIC =             (33, 0)
    TROPIC_THUNDER =    (34, 0.46)
    ZOMBIFYING =        (35, 0.58)

    def __init__(self, idx: int, multiplier: float):
        self.idx = idx
        self.multiplier = multiplier
        self.bitmask = 1 << len(self.__class__.__members__)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name
    
    def print(self):
        print(f"{self.value}) {self.name} = {self.bitmask}")
    
    @staticmethod
    def from_str(effect_name: str) -> Effect:
        effect_name_fixed = effect_name.replace('-', '_').upper()
        try:
            return Effect[effect_name_fixed]
        except KeyError as e:
            print(f"'{effect_name}' ('{effect_name_fixed}') doesn't map to any effects")
            raise e