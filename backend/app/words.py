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


@lru_cache(maxsize=1)
def load() -> Lexicon:
    rows = [line.split("\t") for line in (DATA / "freq.tsv").read_text().splitlines()]
    words = [w for w, _ in rows]
    zipf = np.array([float(z) for _, z in rows])
    plural = np.array([_is_plural(w) for w in words])
    # Smooth step: common words ~1, rare words ~0, plurals heavily discounted.
    prior = 1.0 / (1.0 + np.exp(-2.0 * (zipf - ANSWER_ZIPF)))
    prior = np.where(plural, prior * 0.05, prior)

    letters = np.frombuffer("".join(words).encode(), dtype=np.uint8).reshape(-1, 5) - ord("a")
    answer_pool = np.flatnonzero((zipf >= ANSWER_ZIPF) & ~plural)
    return Lexicon(words, {w: i for i, w in enumerate(words)}, letters, zipf, prior, answer_pool)

