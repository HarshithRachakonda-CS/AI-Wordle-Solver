"""SQLite storage for games, guesses, hint usage and predictor weights."""
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.environ.get("WORDLE_DB", Path(__file__).resolve().parent.parent / "wordle.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    mode       TEXT NOT NULL,          -- 'play' (we pick the answer) or 'assist' (user plays elsewhere)
    answer     TEXT,                   -- NULL in assist mode until solved
    status     TEXT NOT NULL,          -- 'active', 'won', 'lost'
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS guesses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id    TEXT NOT NULL REFERENCES games(id),
    turn       INTEGER NOT NULL,
    word       TEXT NOT NULL,
    pattern    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS hint_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id    TEXT NOT NULL REFERENCES games(id),
    turn       INTEGER NOT NULL,
    level      INTEGER NOT NULL,
    mode       TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS model_weights (
    scope      TEXT PRIMARY KEY,       -- 'global' or a user id
    weights    TEXT NOT NULL,          -- JSON {feature: weight}
    examples   INTEGER NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_games_user ON games(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_guesses_game ON guesses(game_id, turn);
"""


_ready = False


@contextmanager
def connect():
    global _ready
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if not _ready:
        conn.executescript(SCHEMA)
        _ready = True
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with connect():
        pass  # creating the connection creates the tables


def create_game(user_id: str, mode: str, answer: str | None) -> str:
    game_id = uuid.uuid4().hex[:12]
    with connect() as c:
        c.execute("INSERT INTO games VALUES (?, ?, ?, ?, 'active', ?)", (game_id, user_id, mode, answer, time.time()))
    return game_id


def get_game(game_id: str) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        if not row:
            return None
        game = dict(row)
        game["guesses"] = [dict(r) for r in c.execute(
            "SELECT turn, word, pattern FROM guesses WHERE game_id = ? ORDER BY turn", (game_id,))]
        game["hints_used"] = c.execute("SELECT COUNT(*) FROM hint_events WHERE game_id = ?", (game_id,)).fetchone()[0]
    return game


def add_guess(game_id: str, turn: int, word: str, pattern: str, status: str, answer: str | None) -> None:
    with connect() as c:
        c.execute("INSERT INTO guesses (game_id, turn, word, pattern, created_at) VALUES (?, ?, ?, ?, ?)",
                  (game_id, turn, word, pattern, time.time()))
        c.execute("UPDATE games SET status = ?, answer = COALESCE(answer, ?) WHERE id = ?", (status, answer, game_id))


def log_hint(game_id: str, turn: int, level: int, mode: str) -> None:
    with connect() as c:
        c.execute("INSERT INTO hint_events (game_id, turn, level, mode, created_at) VALUES (?, ?, ?, ?, ?)",
                  (game_id, turn, level, mode, time.time()))


def user_word_counts(user_id: str, before: float | None = None) -> tuple[dict[str, int], dict[str, int]]:
    """(every word this user has typed -> count, opening words -> count), optionally only from earlier games."""
    sql = """SELECT g.word, g.turn FROM guesses g JOIN games m ON m.id = g.game_id
             WHERE m.user_id = ? AND m.created_at < ?"""
    typed: dict[str, int] = {}
    openers: dict[str, int] = {}
    with connect() as c:
        for r in c.execute(sql, (user_id, before if before is not None else float("inf"))):
            typed[r["word"]] = typed.get(r["word"], 0) + 1
            if r["turn"] == 1:
                openers[r["word"]] = openers.get(r["word"], 0) + 1
    return typed, openers


