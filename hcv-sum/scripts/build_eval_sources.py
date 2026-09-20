"""Download and convert four REAL public documents into plain text for the labelled pair set.

Usage:  python scripts/build_eval_sources.py

One document per type in the project brief. Everything lands in data/external/ (git-ignored): the
source files are not redistributed, only short quoted sentence pairs end up in data/eval/.

| type      | document                                                        | license / status          |
|-----------|-----------------------------------------------------------------|---------------------------|
| research  | Chen et al. 2025, "On Reference (In-)Determinacy in NLI" (ar5iv) | CC BY-SA 4.0              |
| financial | FOMC minutes, September 17-18, 2024 (federalreserve.gov)        | US government, public domain |
| contract  | ImageWare Systems maintenance agreement, from CUAD v1           | CC BY 4.0                 |
| transcript| FOMC press conference transcript, September 18, 2024            | US government, public domain |

Two substitutions, disclosed: the "financial filing" is a central-bank report, not a corporate 10-K
(SEC EDGAR requires a personal contact e-mail in every request, which was not sent); the "earnings
call" is a central-bank press conference, which has the same opening-statement-then-Q&A shape.
"""

from __future__ import annotations

import html
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "eval_sources"
OUT = ROOT / "data" / "external" / "eval_docs"
UA = {"User-Agent": "HCV-Sum research evaluation (academic, non-commercial)"}

SOURCES = {
    "research_refnli.md": ("https://ar5iv.labs.arxiv.org/html/2502.05793", "paper_refnli.htm"),
    "financial_fomc_minutes.md": ("https://www.federalreserve.gov/monetarypolicy/fomcminutes20240918.htm",
                                  "fomc_minutes.htm"),
    "transcript_fomc_presconf.txt": ("https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20240918.pdf",
                                     "fed_presconf_transcript.pdf"),
    "contract_imageware.txt": ("https://github.com/TheAtticusProject/cuad/raw/main/data.zip", "cuad_data.zip"),
}
CUAD_TITLE = "IMAGEWARESYSTEMSINC_12_20_1999-EX-10.22-MAINTENANCE AGREEMENT"


def fetch(url: str, name: str) -> bytes:
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def strip_tags(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def research(raw: bytes) -> str:
    """ar5iv: h2 sections and h3 subsections become headings; ltx_p paragraphs become text."""
    s = raw.decode("utf-8", errors="replace")
    s = re.sub(r'<math.*?</math>', " ", s, flags=re.S)             # drop MathML
    s = re.sub(r'<cite class="ltx_cite[^"]*">.*?</cite>', " ", s, flags=re.S)
    s = s.split('class="ltx_bibliography"')[0]                      # stop before references
    s = s.split('ltx_title_appendix')[0]                            # and before appendices
    token = re.compile(r'<h[23] class="ltx_title ltx_title_(?:section|subsection)"[^>]*>(.*?)</h[23]>'
                       r'|<p [^>]*class="ltx_p"[^>]*>(.*?)</p>', flags=re.S)   # ar5iv puts id= before class=
    lines = ["# On Reference (In-)Determinacy in Natural Language Inference", ""]
    for heading, para in token.findall(s):
        if heading:
            title = re.sub(r"^\d+(\.\d+)*\s*", "", strip_tags(heading))
            lines += [f"## {title}", ""]
        elif para and len(strip_tags(para)) > 40:
            lines += [strip_tags(para), ""]
    return "\n".join(lines)


def financial(raw: bytes) -> str:
    """FOMC minutes: the six <strong> section titles become headings; admin tail is dropped."""
    s = raw.decode("utf-8", errors="replace")
    s = s[s.find('id="article"'):]
    sections = ["Developments in Financial Markets and Open Market Operations",
                "Staff Review of the Economic Situation", "Staff Review of the Financial Situation",
                "Staff Economic Outlook", "Participants' Views on Current Conditions and the Economic Outlook",
                "Committee Policy Actions"]
    lines = ["# Minutes of the Federal Open Market Committee, September 17-18, 2024", ""]
    for para in re.findall(r"<p>(.*?)</p>", s, flags=re.S):
        text = strip_tags(para)
        if text.startswith(("Voting for this action", "Notation Vote", "Attendance")):
            break
        heading = next((h for h in sections if text.startswith(h)), None)
        if heading:
            lines += [f"## {heading}", ""]
            text = text[len(heading):].strip()
        if len(text) > 40:
            lines += [text, ""]
    return "\n".join(lines)


def transcript(raw: bytes) -> str:
    """Press conference PDF: page furniture removed, speaker turns kept as paragraphs."""
    from pypdf import PdfReader     # dev-only dependency, used for building this data set only

    reader = PdfReader(io.BytesIO(raw))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = re.sub(r"(?m)^.*Chair Powell.s Press Conference.*FINAL.*$", "", text)
    text = re.sub(r"(?m)^\s*Page \d+ of \d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*September 18, 2024\s*$", "", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)                     # re-join hyphenated line breaks
    # Each speaker turn ("CHAIR POWELL.", "MICHELLE SMITH.", "NICK TIMIRAOS.") starts a paragraph.
    text = re.sub(r"\s*\n\s*(?=[A-Z][A-Z .'-]{3,40}\.\s)", "\n\n", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n\n")]
    return "\n\n".join(p for p in paragraphs if len(p) > 20)


def contract(raw: bytes) -> str:
    data = json.loads(zipfile.ZipFile(io.BytesIO(raw)).read("CUADv1.json"))["data"]
    doc = next(d for d in data if d["title"] == CUAD_TITLE)
    return doc["paragraphs"][0]["context"]


BUILDERS = {"research_refnli.md": research, "financial_fomc_minutes.md": financial,
            "transcript_fomc_presconf.txt": transcript, "contract_imageware.txt": contract}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (url, raw_name) in SOURCES.items():
        text = BUILDERS[name](fetch(url, raw_name))
        (OUT / name).write_text(text, encoding="utf-8")
        words = len(text.split())
        print(f"{name:<32} {words:>6} words (~{words / 450:.0f} pages)  <- {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
