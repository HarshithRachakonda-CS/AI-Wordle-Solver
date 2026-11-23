"""Next-input predictor: which word is this user about to type?

Each plausible word gets a feature vector; a linear score per word goes through a
softmax to give a probability. Weights start hand-tuned and can be learned from
the guess log (`train`), first for everyone, then per user once they have played
enough games.
"""
import numpy as np

from .. import hints, storage
from .words import Lexicon, load

FEATURES = ["consistent", "frequency", "information", "typed_before", "usual_opener", "reuses_clues"]

DEFAULT_WEIGHTS = {
    "consistent": 2.0,     # players mostly type words that fit the clues
    "frequency": 5.0,      # ...and common words
    "information": 1.5,    # ...that are good guesses
    "typed_before": 1.0,   # people reuse their own vocabulary
    "usual_opener": 3.0,   # most players open with the same word every game
    "reuses_clues": 1.0,   # keeps known green/yellow letters (hard-mode style)
}

POOL_SIZE = 250
MIN_USER_GAMES = 20        # games before a user gets personal weights


def weights_for(user_id: str | None) -> dict[str, float]:
    return (user_id and storage.load_weights(user_id)) or storage.load_weights("global") or DEFAULT_WEIGHTS


def _pool(lex: Lexicon, alive: np.ndarray, turn: int, prefix: str, typed: dict[str, int]) -> np.ndarray:
    """Words worth scoring: likely answers, good probes, and the user's own words."""
    if prefix:
        rows = np.array([i for i, w in enumerate(lex.words) if w.startswith(prefix)], dtype=int)
        if len(rows) == 0:
            return rows
        alive_set = set(alive.tolist())
        # Keep consistent words first, then the most common, so the pool stays small.
        order = np.lexsort((-lex.zipf[rows], [r not in alive_set for r in rows.tolist()]))
        mine = [lex.index[w] for w in typed if w.startswith(prefix)]
        return np.unique(np.concatenate([rows[order][:POOL_SIZE], np.array(mine, dtype=int)]))

    probs = hints.answer_probs(lex, alive)
    parts = [alive[np.argsort(-probs)[:POOL_SIZE]], np.array([lex.index[w] for w in typed], dtype=int)]
    if turn == 1:
        parts.append(np.argsort(-lex.zipf)[:POOL_SIZE])
        parts.append(np.array([lex.index[h.word] for h in hints.rank([], "probe", 20)[0]]))
    return np.unique(np.concatenate(parts).astype(int))


def features(history: list[tuple[str, str]], user_id: str | None, prefix: str = "",
             typed: dict[str, int] | None = None, openers: dict[str, int] | None = None,
             alive: np.ndarray | None = None) -> tuple[list[str], np.ndarray]:
    """Candidate words and their (n_words, n_features) matrix for the current game state."""
    lex = load()
    if typed is None or openers is None:
        typed, openers = storage.user_word_counts(user_id) if user_id else ({}, {})
    if alive is None:
        alive = hints.candidates(history, lex)
    turn = len(history) + 1
    pool = _pool(lex, alive, turn, prefix.lower(), typed)
    if len(pool) == 0:
        return [], np.zeros((0, len(FEATURES)))

    alive_set = set(alive.tolist())
    probs = hints.answer_probs(lex, alive) if len(alive) else np.array([])
    # Information each pool word would give against the current candidates (0..1).
    info = np.zeros(len(pool))
    if len(alive) > 1:
        remaining = max(hints.bits_left(probs), 1e-9)
        for j, row in enumerate(pool.tolist()):
            info[j] = hints._entropy(lex, row, alive, probs)[0] / remaining
    clue_letters = {g[i] for g, p in history for i in range(5) if p[i] in "GY"}

    words = [lex.words[r] for r in pool.tolist()]
    X = np.zeros((len(pool), len(FEATURES)))
    X[:, 0] = [r in alive_set for r in pool.tolist()]
    X[:, 1] = lex.zipf[pool] / 7.0
    X[:, 2] = np.minimum(info, 1.0)
    X[:, 3] = [np.log1p(typed.get(w, 0)) for w in words]
    X[:, 4] = [np.log1p(openers.get(w, 0)) if turn == 1 else 0.0 for w in words]
    X[:, 5] = [len(clue_letters & set(w)) / len(clue_letters) if clue_letters else 0.0 for w in words]
    return words, X


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def predict(history: list[tuple[str, str]], user_id: str | None, prefix: str = "", limit: int = 5) -> list[dict]:
    words, X = features(history, user_id, prefix)
    if not words:
        return []
    w = weights_for(user_id)
    probs = _softmax(X @ np.array([w[f] for f in FEATURES]))
    top = np.argsort(-probs)[:limit]
    return [{"word": words[i], "probability": round(float(probs[i]), 4),
             "why": _why(dict(zip(FEATURES, X[i])))} for i in top]


def _why(f: dict[str, float]) -> str:
    bits = []
    if f["usual_opener"] > 0:
        bits.append("your usual opener")
    elif f["typed_before"] > 0:
        bits.append("you've typed it before")
    if f["consistent"]:
        bits.append("fits the clues")
    if f["frequency"] > 0.6:
        bits.append("common word")
    if f["information"] > 0.7:
        bits.append("strong guess")
    return ", ".join(bits) or "plausible word"


# ---------------------------------------------------------------- training

def _examples(user_id: str | None) -> list[tuple[np.ndarray, int]]:
    """(feature matrix, index of the word actually typed) for every logged guess.

    User history features only count games played before the one being replayed,
    so the model never sees the future.
    """
    out = []
    for game in storage.user_games(user_id):
        typed, openers = storage.user_word_counts(game["user_id"], before=game["created_at"])
        history: list[tuple[str, str]] = []
        for g in game["guesses"]:
            words, X = features(history, game["user_id"], typed=typed, openers=openers)
            if g["word"] in words:
                out.append((X, words.index(g["word"])))
            history.append((g["word"], g["pattern"]))
    return out


def _accuracy(examples, w: np.ndarray) -> dict[str, float]:
    if not examples:
        return {"top1": 0.0, "top5": 0.0}
    ranks = [int((X @ w > (X @ w)[y]).sum()) for X, y in examples]
    return {"top1": float(np.mean([r < 1 for r in ranks])), "top5": float(np.mean([r < 5 for r in ranks]))}


def train(user_id: str | None = None, epochs: int = 300, lr: float = 0.5, l2: float = 0.05) -> dict:
    """Fit softmax weights by gradient descent on the guess log.

    Global weights are pulled toward the hand-tuned defaults; per-user weights are
    pulled toward the global ones, so little data means small changes.
    """
    scope = user_id or "global"
    anchor_d = DEFAULT_WEIGHTS if user_id is None else weights_for(None)
    if user_id and len(storage.user_games(user_id)) < MIN_USER_GAMES:
        return {"scope": scope, "trained": False,
                "reason": f"needs at least {MIN_USER_GAMES} games for personal weights"}
    examples = _examples(user_id)
    if not examples:
        return {"scope": scope, "trained": False, "reason": "no guesses logged yet"}

    anchor = np.array([anchor_d[f] for f in FEATURES])
    w = anchor.copy()
    before = _accuracy(examples, w)
    for _ in range(epochs):
        grad = l2 * (w - anchor)
        for X, y in examples:
            p = _softmax(X @ w)
            grad += (X.T @ p - X[y]) / len(examples)
        w -= lr * grad
    weights = dict(zip(FEATURES, map(float, w)))
    storage.save_weights(scope, weights, len(examples))
    return {"scope": scope, "trained": True, "examples": len(examples),
            "accuracy_before": before, "accuracy_after": _accuracy(examples, w),
            "weights": {k: round(v, 3) for k, v in weights.items()}}

