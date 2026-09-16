import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingestion.ingest import (
    _is_prose_line,
    _is_section_header,
    chunk_text,
    extract_text,
    restrict_to_english_section,
)


def test_chunk_text_splits_long_text_into_multiple_chunks():
    text = "\n".join(f"Sentence number {i} with some words in it." for i in range(50))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # a little slack for the overlap prefix carried into the next chunk
    assert all(len(c) <= 100 + 20 + 10 for c in chunks)


def test_chunk_text_short_text_produces_one_chunk():
    text = "Just one short sentence."
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    assert chunks == [text]


def test_chunk_text_starts_new_chunk_at_section_header():
    text = "Some intro text about the product.\nTECHNICAL SPECIFICATIONS\nPower: 1100 Watts"
    chunks = chunk_text(text, chunk_size=1000, overlap=50)
    # the header should start a fresh chunk, not merge into the intro text -
    # otherwise the actual spec gets diluted by unrelated preceding content
    assert any(c.startswith("TECHNICAL SPECIFICATIONS") for c in chunks)


def test_is_section_header_detects_all_caps_labels():
    assert _is_section_header("TECHNICAL SPECIFICATIONS")
    assert _is_section_header("FEATURES:")
    assert not _is_section_header("This is a normal sentence.")
    assert not _is_section_header("a")


def test_is_prose_line_detects_long_sentences_not_short_labels():
    assert _is_prose_line("This is a reasonably long sentence with real words in it.")
    assert not _is_prose_line("SHORT")
    assert not _is_prose_line("")


def test_restrict_to_english_section_keeps_english_drops_other_languages():
    text = (
        "\n[EN]\nThis is the English instructions text that is long enough to count as prose.\n"
        "[FR]\nCeci est le texte francais qui est suffisamment long pour compter comme prose.\n"
        "[DE]\nDies ist der deutsche Text der lang genug ist um als Prosa zu gelten."
    )
    result = restrict_to_english_section(text)
    assert "English instructions" in result
    assert "francais" not in result
    assert "deutsche" not in result


def test_restrict_to_english_section_keeps_nonprose_content_regardless_of_language_tag():
    # regression test for the Black Diamond headlamp bug: a language-neutral
    # icon/spec legend physically sitting under a non-English marker must
    # still survive, since it's short label lines, not real prose
    text = (
        "\n[EN]\nThis is the English instructions text that is long enough to count as prose.\n"
        "[FR]\nIPX8\n>1 m\n30 minutes"
    )
    result = restrict_to_english_section(text)
    assert "IPX8" in result


def test_restrict_to_english_section_no_markers_returns_text_unchanged():
    text = "Just a plain document with no language markers at all."
    assert restrict_to_english_section(text) == text


def test_extract_text_strips_markdown_header(tmp_path):
    md_file = tmp_path / "sample.md"
    md_file.write_text(
        "# Title\n\nSource: https://example.com\n\n---\n\nThe real content starts here.",
        encoding="utf-8",
    )
    result = extract_text(md_file)
    assert result.strip() == "The real content starts here."


def test_extract_text_plain_text_file_returned_as_is(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Plain content, nothing special.", encoding="utf-8")
    assert extract_text(txt_file) == "Plain content, nothing special."
