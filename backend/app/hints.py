"""Hint engine: constraint filtering, then entropy + likelihood ranking."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .feedback import pattern_codes, pattern_to_code
from .words import Lexicon, load

# How many guesses to score exactly. Candidates are always included; the rest of
# the pool is filled with words that cover the most common unknown letters.
PROBE_POOL = 400


@dataclass
class Hint:
    word: str
    entropy: float          # expected bits of information from this guess
    p_answer: float         # probability this word is the answer
    groups: int             # how many feedback groups it splits candidates into
    expected_turns: float   # estimated guesses to finish, including this one (lower is better)
    reason: str


# Probe words must be recognisable; nobody wants to be told to type "tarse".
PROBE_MIN_ZIPF = 2.5
# Roughly how many bits a decent guess removes; turns a bit count into turns.
BITS_PER_TURN = 4.0


def candidates(history: list[tuple[str, str]], lex: Lexicon | None = None) -> np.ndarray:
    """Row ids of every word consistent with all (guess, pattern) pairs so far."""
    lex = lex or load()
    alive = np.arange(len(lex.words))
    for guess, pattern in history:
        codes = pattern_codes(lex.encode(guess), lex.letters[alive])
        alive = alive[codes == pattern_to_code(pattern)]
    return alive


def answer_probs(lex: Lexicon, alive: np.ndarray) -> np.ndarray:
    w = lex.prior[alive]
    return w / w.sum() if w.sum() > 0 else np.full(len(alive), 1 / len(alive))


def bits_left(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log2(p)).sum())


def _letter_presence(letters: np.ndarray) -> np.ndarray:
    present = np.zeros((len(letters), 26))
    for i in range(5):
        present[np.arange(len(letters)), letters[:, i]] = 1
    return present


def _probe_pool(lex: Lexicon, alive: np.ndarray, probs: np.ndarray) -> np.ndarray:
    likely = alive[np.argsort(-probs)[:PROBE_POOL]]
    if len(alive) <= 2:
        return likely
    # Weighted share of candidates containing each letter; letters near 50% split best.
    share = probs @ _letter_presence(lex.letters[alive])
    coverage = _letter_presence(lex.letters) @ (share * (1 - share))
    coverage[lex.zipf < PROBE_MIN_ZIPF] = -1
    return np.unique(np.concatenate([np.argsort(-coverage)[:PROBE_POOL], likely]))


def _shot_pool(lex: Lexicon, alive: np.ndarray, probs: np.ndarray) -> np.ndarray:
    plausible = alive[probs >= 0.001]
    pool = plausible if len(plausible) else alive
    return pool[np.argsort(-lex.prior[pool])[:PROBE_POOL]]


def _entropy(lex: Lexicon, guess_row: int, alive: np.ndarray, probs: np.ndarray) -> tuple[float, int]:
    codes = pattern_codes(lex.letters[guess_row], lex.letters[alive])
    mass = np.bincount(codes, weights=probs, minlength=243)
    mass = mass[mass > 0]
    return float(-(mass * np.log2(mass)).sum()), len(mass)


def rank(history: list[tuple[str, str]], mode: str = "probe", limit: int = 5) -> tuple[list[Hint], np.ndarray]:
    """Best next guesses, ordered by fewest expected turns to finish.

    mode="probe": any recognisable word, even one that cannot be the answer.
    mode="shot":  only words that could plausibly be the answer.
    """
    lex = load()
    if not history:
        return list(_opening_hints(mode))[:limit], np.arange(len(lex.words))
    alive = candidates(history, lex)
    if len(alive) == 0:
        return [], alive
    return _score(lex, alive, mode, limit), alive


def _score(lex: Lexicon, alive: np.ndarray, mode: str, limit: int) -> list[Hint]:
    probs = answer_probs(lex, alive)
    p_of = dict(zip(alive.tolist(), probs.tolist()))
    remaining = bits_left(probs)
    pool = _shot_pool(lex, alive, probs) if mode == "shot" else _probe_pool(lex, alive, probs)

    hints = []
    for row in pool.tolist():
        h, groups = _entropy(lex, row, alive, probs)
        p = p_of.get(row, 0.0)
        # Win now with probability p; otherwise one turn spent plus the bits still unknown.
        turns = p * 1 + (1 - p) * (2 + max(remaining - h, 0) / BITS_PER_TURN)
        hints.append(Hint(lex.words[row], h, p, groups, turns, ""))
    hints.sort(key=lambda x: x.expected_turns)
    top = hints[:limit]
    likely = max(round(2 ** remaining), 1)
    for h in top:
        h.reason = _explain(h, likely)
    return top


@lru_cache(maxsize=2)
def _opening_hints(mode: str) -> tuple[Hint, ...]:
    # The first guess never depends on the game, so rank it once and cache it.
    lex = load()
    return tuple(_score(lex, np.arange(len(lex.words)), mode, 20))


def _explain(h: Hint, likely: int) -> str:
    if h.p_answer >= 0.5:
        return f"most likely answer ({h.p_answer:.0%})"
    split = f"splits about {likely} likely words into {h.groups} groups"
    if h.p_answer >= 0.01:
        return f"could be the answer ({h.p_answer:.0%}) and {split}"
    if h.p_answer > 0:
        return f"long shot for the answer, but {split}"
    return f"can't be the answer, but {split}"


def letter_hint(history: list[tuple[str, str]]) -> str | None:
    """The unconfirmed letter most likely to be in the answer."""
    lex = load()
    alive = candidates(history, lex)
    if len(alive) == 0:
        return None
    probs = answer_probs(lex, alive)
    known = {g[i] for g, p in history for i in range(5) if p[i] in "GY"}
    share = np.zeros(26)
    for letter in range(26):
        share[letter] = probs[(lex.letters[alive] == letter).any(axis=1)].sum()
    for letter in np.argsort(-share):
        ch = chr(int(letter) + ord("a"))
        if ch not in known and share[letter] > 0:
            return ch
    return None
