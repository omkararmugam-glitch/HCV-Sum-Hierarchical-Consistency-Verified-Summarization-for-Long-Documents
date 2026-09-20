"""Stage 2 checkpoint/resume: completed section generations are appended to disk as they finish.

Stage 2 is the longest stage on large inputs (~2 s per section on CPU). If a 1,000-section run dies
at section 800, re-running with the same checkpoint directory regenerates only the missing 200.

What is cached is the summarizer's RAW output for one exact model input. The key is a hash of
(model input text, min_new_tokens, max_new_tokens), and the file itself is named by a hash of the
document text plus every setting that can change a generation (summarizer model, Stage 1 and Stage 2
config). So a changed document, model or config never reuses a stale summary: it gets a different
file, or a key miss. Everything after generation (sentence splitting, leak guard, cross-reference
protection) re-runs on resume, so the resumed result is identical to an uninterrupted run
(tests/test_checkpoint.py).

Format: JSON Lines, one {"key", "section_index", "generated"} object per line, flushed after each
batch. A line truncated by a crash mid-write is skipped on load (it is regenerated).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def generation_key(model_input: str, min_new: int, max_new: int) -> str:
    return hashlib.sha256(f"{min_new}|{max_new}|{model_input}".encode("utf-8")).hexdigest()


def checkpoint_path(directory: str | Path, document_text: str, settings: dict) -> Path:
    doc = hashlib.sha256(document_text.encode("utf-8")).hexdigest()[:16]
    conf = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return Path(directory) / f"stage2_{doc}_{conf}.jsonl"


class Stage2Checkpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: dict[str, str] = {}
        self.corrupt_lines = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    self._done[row["key"]] = row["generated"]
                except (json.JSONDecodeError, KeyError):
                    self.corrupt_lines += 1
        self.loaded = len(self._done)
        self.hits = 0
        self.writes = 0

    def get(self, key: str) -> str | None:
        text = self._done.get(key)
        if text is not None:
            self.hits += 1
        return text

    def put_many(self, rows: "list[tuple[str, int, str]]") -> None:
        """Append (key, section_index, generated) rows and flush, so a crash loses at most one batch."""
        with open(self.path, "a", encoding="utf-8") as fh:
            for key, section_index, generated in rows:
                fh.write(json.dumps({"key": key, "section_index": section_index, "generated": generated},
                                    ensure_ascii=False) + "\n")
                self._done[key] = generated
                self.writes += 1
            fh.flush()
