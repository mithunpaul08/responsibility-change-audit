"""Local reader LLM via mlx-lm, with an on-disk answer cache.

The cache (results/cache/reader.sqlite) is keyed by sha256(model, prompt,
max_tokens). Re-running any experiment therefore costs nothing for prompts
already seen, and two methods that produce the same prompt share one call.

Decoding is greedy (temperature 0) so a given prompt always yields the same
answer; this is required for the `agree_full` metric to be meaningful.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "results" / "cache" / "reader.sqlite"

DEFAULT_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"
SMOKE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

SYSTEM_PROMPT = ("You answer questions using only the given context. "
                 "Reply with the answer only: a single word or short phrase on one line, "
                 "with no explanation.")


def build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


class ReaderCache:
    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS gen (key TEXT PRIMARY KEY, model TEXT, "
                          "prompt TEXT, output TEXT, prompt_tokens INT, latency REAL, ts REAL)")
        self.conn.commit()

    @staticmethod
    def key(model: str, prompt: str, max_tokens: int) -> str:
        return hashlib.sha256(f"{model}\x00{max_tokens}\x00{prompt}".encode()).hexdigest()

    def get(self, key: str):
        row = self.conn.execute("SELECT output, prompt_tokens FROM gen WHERE key=?", (key,)).fetchone()
        return row

    def put(self, key: str, model: str, prompt: str, output: str, ptoks: int, latency: float):
        self.conn.execute("INSERT OR REPLACE INTO gen VALUES (?,?,?,?,?,?,?)",
                          (key, model, prompt, output, ptoks, latency, time.time()))
        self.conn.commit()


class MLXReader:
    def __init__(self, model_id: str = DEFAULT_MODEL, max_tokens: int = 16,
                 cache: ReaderCache | None = None):
        from mlx_lm import load
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.model, self.tokenizer = load(model_id)
        self.cache = cache or ReaderCache()
        self._sampler = None
        try:  # mlx-lm >= 0.20 uses a sampler object for temperature
            from mlx_lm.sample_utils import make_sampler
            self._sampler = make_sampler(temp=0.0)
        except Exception:  # noqa: BLE001
            pass

    # -- tokens ------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _chat(self, user: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # -- generation --------------------------------------------------------
    def generate(self, prompt_user: str) -> tuple[str, int]:
        """Returns (raw generation, prompt token count). Cached."""
        from mlx_lm import generate
        full = self._chat(prompt_user)
        k = self.cache.key(self.model_id, full, self.max_tokens)
        hit = self.cache.get(k)
        if hit is not None:
            return hit[0], hit[1]
        t0 = time.time()
        kwargs = dict(max_tokens=self.max_tokens, verbose=False)
        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
        else:
            kwargs["temp"] = 0.0
        out = generate(self.model, self.tokenizer, prompt=full, **kwargs)
        ptoks = self.count_tokens(full)
        self.cache.put(k, self.model_id, full, out, ptoks, time.time() - t0)
        return out, ptoks

    def answer(self, context: str, question: str) -> tuple[str, int]:
        return self.generate(build_prompt(context, question))
