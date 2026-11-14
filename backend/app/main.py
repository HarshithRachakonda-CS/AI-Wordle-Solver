"""HTTP API for the AI Wordle Helper."""
import random
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import hints, predictor, storage
from .feedback import score
from .words import load

MAX_TURNS = 6


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init()
    load()
    # Opening-move rankings take a few seconds; compute them off the request path.
    threading.Thread(target=lambda: [hints.rank([], m) for m in ("probe", "shot")], daemon=True).start()
    yield


app = FastAPI(title="AI Wordle Helper", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ models

Word = Annotated[str, Field(pattern="^[A-Za-z]{5}$")]
Pattern = Annotated[str, Field(pattern="^[GYBgyb]{5}$", description="G = right spot, Y = wrong spot, B = not in word")]


class NewGame(BaseModel):
    user_id: str = "anonymous"
    mode: Literal["play", "assist"] = "play"


class GuessIn(BaseModel):
    word: Word
    pattern: Pattern | None = None  # required in assist mode: the colours the real game showed


class GuessFeedback(BaseModel):
    word: Word
    pattern: Pattern


class SolveIn(BaseModel):
    guesses: list[GuessFeedback] = []
    mode: Literal["probe", "shot"] = "probe"
    limit: int = Field(5, ge=1, le=20)


# ------------------------------------------------------------------ helpers

def _game_or_404(game_id: str) -> dict:
    game = storage.get_game(game_id)
    if not game:
        raise HTTPException(404, "game not found")
    return game


def _history(game: dict) -> list[tuple[str, str]]:
    return [(g["word"], g["pattern"]) for g in game["guesses"]]


def _public(game: dict) -> dict:
    """Game state safe to send to the player (answer hidden until the game ends)."""
    lex = load()
    alive = hints.candidates(_history(game), lex)
    likely = max(round(2 ** hints.bits_left(hints.answer_probs(lex, alive))), 1) if len(alive) else 0
    return {
        "game_id": game["id"], "user_id": game["user_id"], "mode": game["mode"], "status": game["status"],
        "turn": len(game["guesses"]) + 1, "max_turns": MAX_TURNS, "guesses": game["guesses"],
        "hints_used": game["hints_used"], "words_left": int(len(alive)), "likely_words_left": likely,
        "answer": game["answer"] if game["status"] != "active" else None,
    }


def _hint_rows(items: list[hints.Hint]) -> list[dict]:
    return [{"word": h.word, "p_answer": round(h.p_answer, 4), "entropy_bits": round(h.entropy, 2),
             "expected_turns": round(h.expected_turns, 2), "reason": h.reason} for h in items]


# ------------------------------------------------------------------ routes

@app.get("/health")
def health():
    return {"ok": True, "words": len(load().words)}


@app.post("/games", status_code=201)
def new_game(body: NewGame):
    lex = load()
    answer = None
    if body.mode == "play":
        answer = lex.words[int(random.choice(lex.answer_pool))]
    game_id = storage.create_game(body.user_id, body.mode, answer)
    return _public(_game_or_404(game_id))


@app.get("/games/{game_id}")
def get_game(game_id: str):
    return _public(_game_or_404(game_id))


@app.post("/games/{game_id}/guesses")
def guess(game_id: str, body: GuessIn):
    game = _game_or_404(game_id)
    if game["status"] != "active":
        raise HTTPException(409, f"game already {game['status']}")
    word = body.word.lower()
    if not load().is_valid(word):
        raise HTTPException(422, f"'{word}' is not in the word list")

    if game["mode"] == "play":
        pattern = score(word, game["answer"])
    elif body.pattern:
        pattern = body.pattern.upper()
    else:
        raise HTTPException(422, "assist mode needs the pattern the real game showed")

    turn = len(game["guesses"]) + 1
    status = "won" if pattern == "GGGGG" else "lost" if turn >= MAX_TURNS else "active"
    storage.add_guess(game_id, turn, word, pattern, status, word if status == "won" else None)
    return {"feedback": pattern, **_public(_game_or_404(game_id))}


@app.get("/games/{game_id}/hints")
def get_hints(game_id: str,
              level: int = Query(3, ge=1, le=4, description="1 count, 2 letter, 3 best guesses, 4 likely answer"),
              mode: Literal["probe", "shot"] = "probe", limit: int = Query(5, ge=1, le=20)):
    game = _game_or_404(game_id)
    if game["status"] != "active":
        raise HTTPException(409, f"game already {game['status']}")
    history = _history(game)
    storage.log_hint(game_id, len(history) + 1, level, mode)
    state = _public(game)

    if level == 1:
        return {"level": 1, "message": f"About {state['likely_words_left']} likely words are still possible."}
    if level == 2:
        letter = hints.letter_hint(history)
        return {"level": 2, "letter": letter, "message": f"The answer probably contains '{letter}'."}
    if level == 3:
        ranked, _ = hints.rank(history, mode, limit)
        return {"level": 3, "mode": mode, "suggestions": _hint_rows(ranked)}
    ranked, _ = hints.rank(history, "shot", limit)
    best = max(ranked, key=lambda h: h.p_answer) if ranked else None
    return {"level": 4, "most_likely_answer": best.word if best else None,
            "p_answer": round(best.p_answer, 4) if best else 0, "suggestions": _hint_rows(ranked)}


@app.get("/games/{game_id}/autocomplete")
def autocomplete(game_id: str, prefix: str = Query(..., min_length=1, max_length=5, pattern="^[A-Za-z]+$"),
                 limit: int = Query(5, ge=1, le=20)):
    game = _game_or_404(game_id)
    return {"prefix": prefix.lower(),
            "suggestions": predictor.predict(_history(game), game["user_id"], prefix, limit)}


@app.get("/games/{game_id}/predict")
def predict_next(game_id: str, limit: int = Query(5, ge=1, le=20)):
    game = _game_or_404(game_id)
    return {"turn": len(game["guesses"]) + 1,
            "predictions": predictor.predict(_history(game), game["user_id"], "", limit)}


@app.post("/solve")
def solve(body: SolveIn):
    """Stateless hints for a game played elsewhere: send the guesses and colours so far."""
    history = [(g.word.lower(), g.pattern.upper()) for g in body.guesses]
    ranked, alive = hints.rank(history, body.mode, body.limit)
    return {"words_left": int(len(alive)), "suggestions": _hint_rows(ranked)}


@app.get("/users/{user_id}/stats")
def stats(user_id: str):
    games = [g for g in storage.user_games(user_id) if g["status"] != "active"]
    wins = [g for g in games if g["status"] == "won"]
    _, openers = storage.user_word_counts(user_id)
    return {
        "user_id": user_id, "games": len(games), "wins": len(wins),
        "win_rate": round(len(wins) / len(games), 3) if games else None,
        "avg_turns_to_win": round(sum(len(g["guesses"]) for g in wins) / len(wins), 2) if wins else None,
        "avg_hints_per_game": round(sum(g["hints_used"] for g in games) / len(games), 2) if games else None,
        "favourite_openers": sorted(openers.items(), key=lambda kv: -kv[1])[:5],
        "personal_model": storage.load_weights(user_id) is not None,
    }


@app.post("/model/train")
def train(user_id: str | None = None):
    """Retrain the next-word predictor from logged guesses (everyone, or one user)."""
    return predictor.train(user_id)
