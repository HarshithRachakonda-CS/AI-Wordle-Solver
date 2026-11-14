"""Wordle scoring rules.

A feedback pattern is a 5-character string of G (right letter, right spot),
Y (in the word, wrong spot) and B (not in the word, or no copies left).
Duplicate letters follow the official rules: greens are claimed first, then
yellows left to right while unclaimed copies of that letter remain.
"""
from collections import Counter

import numpy as np

GREEN, YELLOW, BLACK = 2, 1, 0
_CHARS = {GREEN: "G", YELLOW: "Y", BLACK: "B"}
_POW3 = 3 ** np.arange(5)


def score(guess: str, answer: str) -> str:
    """Feedback for one guess against one answer, e.g. score("speed", "abide") == "BBYBY"."""
    result = ["B"] * 5
    unclaimed = Counter(a for g, a in zip(guess, answer) if g != a)
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            result[i] = "G"
        elif unclaimed[g] > 0:
            result[i] = "Y"
            unclaimed[g] -= 1
    return "".join(result)


def pattern_codes(guess: np.ndarray, answers: np.ndarray) -> np.ndarray:
    """Vectorised `score` of one encoded guess (5,) against many answers (N, 5).

    Returns an int code 0..242 per answer (base-3 digits, GREEN=2 YELLOW=1 BLACK=0)
    so patterns can be bucketed with np.bincount.
    """
    green = answers == guess                                   # (N, 5)
    # For each distinct guess letter, how many copies sit in the answer's non-green spots.
    unclaimed = {int(l): ((answers == l) & ~green).sum(axis=1) for l in set(guess.tolist())}

    marks = np.where(green, GREEN, BLACK).astype(np.int16)
    for i in range(5):
        left = unclaimed[int(guess[i])]
        hit = ~green[:, i] & (left > 0)
        marks[hit, i] = YELLOW
        left[hit] -= 1
    return marks @ _POW3


def code_to_pattern(code: int) -> str:
    return "".join(_CHARS[(code // 3**i) % 3] for i in range(5))


def pattern_to_code(pattern: str) -> int:
    lookup = {"G": GREEN, "Y": YELLOW, "B": BLACK}
    return sum(lookup[c] * 3**i for i, c in enumerate(pattern.upper()))
