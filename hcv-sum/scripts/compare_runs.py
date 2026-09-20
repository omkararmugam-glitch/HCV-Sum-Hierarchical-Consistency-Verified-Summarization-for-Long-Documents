"""Check that two runs of the sample documents produced IDENTICAL results.

Usage:  python scripts/compare_runs.py outputs/<baseline_tag> outputs/<candidate_tag>

Compares, per document JSON: segmentation, every section summary, Stage 3a / 3b / diagnostic flags
(claims, scores, resolutions), corrected sentences, merge mode and final summary, and the evidence
panel (status, entailment, citation, flags). Ignored on purpose: wall-clock timings, memory, and stats
keys that did not exist in the baseline (new instrumentation cannot be "different" from nothing).
Scores are compared exactly; any difference is printed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

IGNORED_TOP = {"timings", "config"}
IGNORED_STATS = {"seconds_total", "per_stage"}


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        yield f"{prefix}#len", len(obj)
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def comparable(run: dict, baseline_stats_keys: set[str]) -> dict:
    out = {k: v for k, v in run.items() if k not in IGNORED_TOP}
    stats = out.get("stats") or {}
    out["stats"] = {k: v for k, v in stats.items() if k in baseline_stats_keys and k not in IGNORED_STATS}
    return out


def main() -> int:
    base_dir, cand_dir = Path(sys.argv[1]), Path(sys.argv[2])
    all_same = True
    for base_file in sorted(base_dir.glob("*.json")):
        cand_file = cand_dir / base_file.name
        if not cand_file.exists():
            print(f"MISSING in candidate: {base_file.name}")
            all_same = False
            continue
        base = json.loads(base_file.read_text(encoding="utf-8"))
        cand = json.loads(cand_file.read_text(encoding="utf-8"))
        keys = set((base.get("stats") or {}).keys())
        a, b = dict(flatten(comparable(base, keys))), dict(flatten(comparable(cand, keys)))
        diffs = [k for k in sorted(set(a) | set(b)) if a.get(k, "<absent>") != b.get(k, "<absent>")]
        if diffs:
            all_same = False
            print(f"DIFFERENT: {base_file.name} ({len(diffs)} fields)")
            for k in diffs[:15]:
                print(f"    {k}: {a.get(k, '<absent>')!r:.90} -> {b.get(k, '<absent>')!r:.90}")
        else:
            print(f"IDENTICAL: {base_file.name} ({len(a)} fields compared)")
    print("\nALL IDENTICAL" if all_same else "\nDIFFERENCES FOUND")
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
