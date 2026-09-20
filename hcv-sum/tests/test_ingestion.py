"""File ingestion (src/hcv_sum/ingestion.py): every supported format becomes the plain text Stage 1 expects;
everything else fails before any model is loaded.

PDFs are built here from scratch (tiny valid PDFs with a Helvetica text layer), DOCX files with python-docx,
so no binary fixtures are checked in and every expected string is visible in this file."""

import json

import pytest

from hcv_sum import cli, ingestion, multidoc as multidoc_module
from hcv_sum.ingestion import IngestionError, clean_pdf_pages, ingest
from hcv_sum.models import ModelRegistry

from conftest import FakeEmbedder, FakeNLI, FakeSummarizer


# ------------------------------------------------------------------------------------------ fixtures
def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    """A minimal valid PDF: one page per list, one text line per string (top to bottom), Helvetica 11pt.
    An empty list gives a page with no text at all (what a scanned page looks like to a text extractor)."""
    n = len(pages)
    font_id = 3 + 2 * n
    objects = {1: "<< /Type /Catalog /Pages 2 0 R >>",
               2: f"<< /Type /Pages /Kids [{' '.join(f'{3 + 2 * i} 0 R' for i in range(n))}] /Count {n} >>",
               font_id: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    streams = {}
    for i, lines in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        objects[page_id] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>")
        ops = "".join(f"BT /F1 11 Tf 72 {750 - 16 * j} Td ({_escape(line)}) Tj ET\n" for j, line in enumerate(lines))
        streams[content_id] = ops.encode("latin-1")
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for obj_id in range(1, font_id + 1):
        offsets[obj_id] = len(out)
        if obj_id in streams:
            data = streams[obj_id]
            out += f"{obj_id} 0 obj\n<< /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream\nendobj\n"
        else:
            out += f"{obj_id} 0 obj\n{objects[obj_id]}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {font_id + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offsets[i]:010d} 00000 n \n" for i in range(1, font_id + 1)).encode()
    out += f"trailer\n<< /Size {font_id + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


BODY = [
    ["The committee reviewed the regional water budget for the coming year.",
     "Reservoir levels were eleven percent below the ten-year average in March."],
    ["Groundwater withdrawals rose in the northern districts during the drought.",
     "The utility proposed a tiered tariff to discourage peak summer use."],
    ["Two treatment plants will be upgraded before the end of the fiscal year.",
     "The upgrade is funded by a state grant and a municipal bond issue."],
    ["Public hearings on the tariff are scheduled for the autumn.",
     "The committee will publish its final recommendations in December."],
    ["Leak detection reduced distribution losses by four percent last year.",
     "Further metering work is planned for the older eastern network."],
]


def report_pdf(tmp_path, name="report.pdf"):
    """Five pages, each with a running header, a running footer and a page-number line around the body."""
    pages = [["Regional Water Authority - Annual Budget Review", "CONFIDENTIAL DRAFT", *body,
              "Prepared by the Finance Office", f"Page {i + 1} of {len(BODY)}"] for i, body in enumerate(BODY)]
    path = tmp_path / name
    path.write_bytes(make_pdf(pages))
    return path


def make_docx(path, paragraphs):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    for style, text in paragraphs:
        if style == "Title":
            document.add_heading(text, level=0)
        elif style.startswith("Heading"):
            document.add_heading(text, level=int(style.split()[1]))
        else:
            document.add_paragraph(text)
    document.save(str(path))
    return path


# ------------------------------------------------------------------------------------------ routing
@pytest.mark.parametrize("name,handler", [("a.txt", "_from_text"), ("a.md", "_from_text"), ("a.MD", "_from_text"),
                                          ("a.pdf", "_from_pdf"), ("a.json", "_from_json"), ("a.docx", "_from_docx"),
                                          ("a.PDF", "_from_pdf")])
def test_each_extension_routes_to_its_handler(tmp_path, monkeypatch, name, handler):
    calls = []
    for h in ("_from_text", "_from_pdf", "_from_json", "_from_docx"):
        monkeypatch.setattr(ingestion, h, lambda path, data, *rest, _h=h: calls.append(_h) or "sentinel")
    path = tmp_path / name
    path.write_bytes(b"x")
    assert ingest(path) == "sentinel"
    assert calls == [handler]


# ------------------------------------------------------------------------------------------ unsupported input
@pytest.mark.parametrize("name", ["slides.pptx", "data.xlsx", "photo.png", "program.exe", "README", "archive.tar.gz"])
def test_unsupported_extensions_fail_naming_the_format_and_listing_supported(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
    with pytest.raises(IngestionError) as exc:
        ingest(path)
    msg = str(exc.value)
    assert "unsupported input format" in msg
    assert ("no file extension" in msg) if name == "README" else (f"'{path.suffix}'" in msg)
    for ext in (".txt", ".md", ".pdf", ".json", ".docx"):
        assert ext in msg


def test_content_that_does_not_match_the_extension_is_rejected(tmp_path):
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("just some text pretending to be a PDF", encoding="utf-8")
    with pytest.raises(IngestionError, match="does not look like a PDF"):
        ingest(fake_pdf)
    fake_docx = tmp_path / "fake.docx"
    fake_docx.write_text("plain text", encoding="utf-8")
    with pytest.raises(IngestionError, match="does not look like a .docx"):
        ingest(fake_docx)
    binary_txt = tmp_path / "binary.txt"
    binary_txt.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00")
    with pytest.raises(IngestionError, match="binary data"):
        ingest(binary_txt)


def test_cli_rejects_an_unsupported_file_before_loading_any_model(tmp_path, monkeypatch, capsys):
    def no_models(*a, **k):
        raise AssertionError("a model was loaded for an unsupported file")
    monkeypatch.setattr(multidoc_module, "ModelRegistry", no_models)
    good, bad = tmp_path / "a.md", tmp_path / "b.xlsx"
    good.write_text("# Title\n\nSome text here.\n", encoding="utf-8")
    bad.write_bytes(b"PK\x03\x04 spreadsheet")
    assert cli.main([str(good), str(bad)]) == 2
    err = capsys.readouterr().err
    assert "b.xlsx" in err and "unsupported input format ('.xlsx')" in err


# ------------------------------------------------------------------------------------------ .txt / .md
@pytest.mark.parametrize("name", ["doc.txt", "doc.md"])
def test_text_passthrough_is_identical_to_the_previous_cli_read(tmp_path, name):
    raw = "# Heading\n\nFirst paragraph with “curly quotes” and é.\n\n## Next\nTabs\tand  spaces  kept.   \n"
    path = tmp_path / name
    path.write_bytes(raw.encode("utf-8"))
    result = ingest(path)
    assert result.text == path.read_text(encoding="utf-8-sig")      # exactly what cli.py used to do
    assert result.text.encode("utf-8") == raw.encode("utf-8")        # byte-identical to the file
    assert result.format == ("markdown" if name.endswith(".md") else "text")


def test_text_passthrough_drops_only_a_leading_bom(tmp_path):
    path = tmp_path / "bom.md"
    path.write_bytes(b"\xef\xbb\xbf# Title\n\nBody.\n")
    assert ingest(path).text == "# Title\n\nBody.\n"


def test_non_utf8_text_keeps_the_old_error_message(tmp_path):
    path = tmp_path / "latin.txt"
    path.write_bytes("caf\xe9 cr\xe8me".encode("latin-1"))
    with pytest.raises(IngestionError, match="is not UTF-8 text"):
        ingest(path)


# ------------------------------------------------------------------------------------------ .pdf
def test_pdf_strips_repeated_headers_footers_and_page_numbers(tmp_path):
    pytest.importorskip("pypdf")
    result = ingest(report_pdf(tmp_path))
    assert result.format == "pdf" and result.details["pages"] == 5
    for body in BODY:
        for line in body:
            assert line in result.text, line
    for boilerplate in ("Regional Water Authority", "CONFIDENTIAL DRAFT", "Prepared by the Finance Office", "Page 3 of 5"):
        assert boilerplate not in result.text, boilerplate
    assert result.details["repeated_header_footer_lines"] == 15    # 3 repeated lines x 5 pages
    assert result.details["page_number_lines"] == 5
    assert "Ingested as PDF: extracted" in result.summary and "from 5 pages" in result.summary
    # body order is page order
    positions = [result.text.index(body[0]) for body in BODY]
    assert positions == sorted(positions)


def test_pdf_cleaning_rules_on_page_text():
    pages = [f"Acme Corp Quarterly\n{i + 1}\nBody sentence number {i} about pumps.\niv\nFooter text here {i + 1}"
             for i in range(4)]
    text, info = clean_pdf_pages(pages)
    assert "Acme Corp Quarterly" not in text and "Footer text here" not in text
    # Regression: body lines that differ only in a number used to be matched digit-insensitively and deleted
    # as a "running header"; digits are ignored only for short, non-sentence lines now.
    assert all(f"Body sentence number {i} about pumps." in text for i in range(4))
    assert info["page_number_lines"] == 8          # "1".."4" and "iv" on each page
    # a line that merely repeats on two of four pages is NOT treated as boilerplate
    text, _ = clean_pdf_pages(["Shared line\nA.", "Shared line\nB.", "Other\nC.", "Other two\nD."])
    assert text.count("Shared line") == 2


def test_pdf_page_number_patterns():
    for line in ("42", "  7  ", "Page 42", "page 42 of 187", "PAGE 3", "12 / 40", "xii", "iv", "xxxix"):
        assert ingestion._PAGE_NUMBER.match(line), line
    # Regression: the roman-numeral branch was case-insensitive over [ivxlcdm], which took words and headings.
    for line in ("42 percent of wells", "Page layout was revised", "In 2026 the plant", "Mix", "Civil", "DC", "IV",
                 "mix", "did", "Page", "vv", ""):
        assert not ingestion._PAGE_NUMBER.match(line), line


def test_pdf_hyphenated_line_breaks_are_rejoined():
    text, info = clean_pdf_pages(["The treat-\nment plant is new.\nWell-known results stay hyphenated."])
    assert "treatment plant" in text and "Well-known" in text and info["hyphenated_breaks_rejoined"] == 1


def test_scanned_pdf_without_a_text_layer_fails_clearly(tmp_path):
    pytest.importorskip("pypdf")
    path = tmp_path / "scan.pdf"
    path.write_bytes(make_pdf([[], [], []]))       # three pages, no text operators at all
    with pytest.raises(IngestionError) as exc:
        ingest(path)
    msg = str(exc.value)
    assert "no usable text layer" in msg and "OCR" in msg and "not supported" in msg


def test_blank_pdf_from_pypdf_is_also_detected_as_scanned(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    for _ in range(2):
        writer.add_blank_page(width=612, height=792)
    path = tmp_path / "blank.pdf"
    with open(path, "wb") as fh:
        writer.write(fh)
    with pytest.raises(IngestionError, match="OCR"):
        ingest(path)


def test_corrupt_pdf_fails_clearly(tmp_path):
    pytest.importorskip("pypdf")
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\n" + b"\x00garbage" * 50)
    with pytest.raises(IngestionError):
        ingest(path)


# ------------------------------------------------------------------------------------------ .json
def write_json(tmp_path, obj, name="doc.json"):
    path = tmp_path / name
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


@pytest.mark.parametrize("key", ["text", "document"])
def test_json_single_text_field(tmp_path, key):
    result = ingest(write_json(tmp_path, {key: "# Report\n\nBody text.", "author": "ignored"}))
    assert result.text == "# Report\n\nBody text." and result.details["shape"] == key


def test_json_sections_of_strings_and_titled_objects(tmp_path):
    result = ingest(write_json(tmp_path, {"title": "Water Review", "sections": [
        "An untitled opening section.",
        {"title": "Budget", "text": "The budget grew."},
        {"text": "A section without a title."},
        {"title": "Empty", "text": "   "},
    ]}))
    assert result.text == ("# Water Review\n\nAn untitled opening section.\n\n## Budget\n\nThe budget grew.\n\n"
                           "A section without a title.")
    assert result.details["sections"] == 3 and result.details["titled"] == 1


@pytest.mark.parametrize("obj,fragment", [
    ([1, 2, 3], "top level is a list"),
    ({"body": "text"}, "none of the fields"),
    ({"text": "a", "sections": ["b"]}, "more than one of"),
    ({"text": ""}, "non-empty string"),
    ({"text": 42}, "non-empty string"),
    ({"sections": []}, "non-empty array"),
    ({"sections": [{"title": "No text key"}]}, "sections[0] is neither"),
    ({"sections": [["nested", "list"]]}, "sections[0] is neither"),
    ({"sections": [{"title": 5, "text": "x"}]}, "sections[0].title must be a string"),
])
def test_json_unrecognised_shapes_name_the_expected_shapes(tmp_path, obj, fragment):
    with pytest.raises(IngestionError) as exc:
        ingest(write_json(tmp_path, obj))
    msg = str(exc.value)
    assert fragment in msg
    assert '{"text": "<document text>"}' in msg and '"sections"' in msg


def test_invalid_json_fails_clearly(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{'text': 'single quotes are not JSON'}", encoding="utf-8")
    with pytest.raises(IngestionError, match="not valid UTF-8 JSON"):
        ingest(path)


# ------------------------------------------------------------------------------------------ .docx
def test_docx_headings_become_markdown_headings(tmp_path):
    path = make_docx(tmp_path / "doc.docx", [
        ("Title", "Annual Water Review"),
        ("Heading 1", "Budget"),
        ("Normal", "The budget grew by six percent."),
        ("Heading 2", "Capital works"),
        ("Normal", "Two plants will be upgraded."),
        ("Normal", ""),
    ])
    result = ingest(path)
    assert result.text == ("# Annual Water Review\n\n## Budget\n\nThe budget grew by six percent.\n\n"
                           "### Capital works\n\nTwo plants will be upgraded.")
    assert result.details["headings"] == 3 and result.details["tables_skipped"] == 0


def test_docx_tables_are_reported_as_skipped_not_silently_dropped(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Body paragraph.")
    document.add_table(rows=2, cols=2).cell(0, 0).text = "cell text"
    path = tmp_path / "t.docx"
    document.save(str(path))
    result = ingest(path)
    assert "cell text" not in result.text
    assert result.details["tables_skipped"] == 1 and "1 table(s) NOT extracted" in result.summary


def test_docx_segmentation_sees_the_headings(tmp_path):
    from conftest import make_cfg
    from hcv_sum.segmentation import _structured_blocks
    path = make_docx(tmp_path / "s.docx", [("Heading 1", "Budget"), ("Normal", "The budget grew. Costs fell."),
                                           ("Heading 1", "Staff"), ("Normal", "Hiring slowed. Training grew.")])
    blocks, n_headings = _structured_blocks(ingest(path).text, make_cfg().segmentation)
    assert n_headings == 2
    assert [title for title, body in blocks if body.strip()] == ["Budget", "Staff"]


# ------------------------------------------------------------------------------------------ CLI integration
def fake_registry(monkeypatch):
    monkeypatch.setattr(multidoc_module, "ModelRegistry",
                        lambda cfg: ModelRegistry(cfg, embedder=FakeEmbedder(), nli=FakeNLI(), summarizer=FakeSummarizer()))


def test_cli_multi_document_run_with_mixed_formats_logs_each_ingestion(tmp_path, monkeypatch, capsys):
    fake_registry(monkeypatch)
    md = tmp_path / "notes.md"
    md.write_text("# Notes\n\n## Tariff\nThe tariff hearings are in the autumn.\n", encoding="utf-8")
    js = write_json(tmp_path, {"sections": [{"title": "Leaks", "text": "Leak detection cut losses by four percent."}]})
    pdf = report_pdf(tmp_path)
    assert cli.main([str(md), str(js), str(pdf), "--verbose", "--set", "summarization.passthrough_tokens=1000"]) == 0
    err = capsys.readouterr().err
    assert "[notes.md] Ingested as Markdown" in err
    assert "[doc.json] Ingested as JSON (\"sections\" array)" in err
    assert "[report.pdf] Ingested as PDF: extracted" in err and "stripped 15 repeated header/footer lines" in err


def test_cli_single_pdf_reaches_the_pipeline_as_cleaned_text(tmp_path, monkeypatch):
    seen = {}

    class Spy:
        def __init__(self, cfg):
            pass

        def run(self, text, name, **kwargs):
            seen["text"], seen["name"] = text, name
            raise SystemExit(0)
    monkeypatch.setattr("hcv_sum.pipeline.HCVSumPipeline", Spy)
    with pytest.raises(SystemExit):
        cli.main([str(report_pdf(tmp_path))])
    assert seen["name"] == "report.pdf" and BODY[0][0] in seen["text"] and "CONFIDENTIAL" not in seen["text"]


def test_cli_quiet_mode_prints_no_ingestion_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hcv_sum.pipeline.HCVSumPipeline", lambda cfg: (_ for _ in ()).throw(SystemExit(0)))
    path = tmp_path / "a.md"
    path.write_text("# A\n\nText.\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main([str(path)])
    assert "Ingested" not in capsys.readouterr().err
