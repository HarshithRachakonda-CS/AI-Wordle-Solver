"""Word list, frequency prior and the numpy encodings used by the solver."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"

# Words at or above this Zipf frequency are treated as plausible answers.
ANSWER_ZIPF = 3.0


@dataclass(frozen=True)
class Lexicon:
    words: list[str]           # every valid guess, sorted
    index: dict[str, int]      # word -> row
    letters: np.ndarray        # (N, 5) uint8, a=0..z=25
    zipf: np.ndarray           # (N,) float, corpus frequency on the Zipf scale
    prior: np.ndarray          # (N,) float, unnormalised P(word is the answer)
    answer_pool: np.ndarray    # row ids of words a new game may pick as its answer

    def encode(self, word: str) -> np.ndarray:
        return np.frombuffer(word.encode(), dtype=np.uint8) - ord("a")

    def is_valid(self, word: str) -> bool:
        return word in self.index


def _is_plural(word: str) -> bool:
    # Real Wordle answers are almost never simple plurals ("words", "cats").
    # Keep endings that are usually not plurals: glass, bonus, basis.
    return word.endswith("s") and not word.endswith(("ss", "us", "is"))




