"""Cut the first N words of a document, at a paragraph boundary, for a quick first-signal run.

Usage:  python scripts/slice_document.py report.txt report_first25p.txt --words 11500

At ~450 words per printed page, 11,500 words is about 25 pages. The cut is made at the end of the paragraph
in which the word limit is reached, so no sentence or heading is split. Prints what was kept.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("target", type=Path)
    ap.add_argument("--words", type=int, default=11500)
    args = ap.parse_args()
    text = args.source.read_text(encoding="utf-8-sig")
    paragraphs = re.split(r"(\n\s*\n)", text)
    kept, count = [], 0
    for part in paragraphs:
        kept.append(part)
        count += len(part.split())
        if count >= args.words and not part.strip() == "":
            break
    out = "".join(kept).rstrip() + "\n"
    args.target.write_text(out, encoding="utf-8")
    total = len(text.split())
    print(f"kept {len(out.split()):,} of {total:,} words (~{len(out.split()) / 450:.0f} of ~{total / 450:.0f} pages) "
          f"-> {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
