"""Unit tests for text-preprocessing utilities in tortoise.py.

These tests are intentionally dependency-free (no torch / tortoise
required) so they can run in any CI environment.
"""

import pytest

from tts_pipeline.tortoise import _normalize_text, _split_into_chunks, _strip_markdown


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_strips_utf8_bom(self):
        assert _normalize_text("\ufeffHello") == "Hello"

    def test_collapses_internal_whitespace(self):
        assert _normalize_text("hello   world\n\nfoo\tbar") == "hello world foo bar"

    def test_strips_leading_and_trailing(self):
        assert _normalize_text("  hello world  ") == "hello world"

    def test_empty_string(self):
        assert _normalize_text("") == ""

    def test_whitespace_only(self):
        assert _normalize_text("   \n\t  ") == ""

    def test_single_word(self):
        assert _normalize_text("word") == "word"

    def test_bom_plus_whitespace(self):
        assert _normalize_text("\ufeff  \n hello \n") == "hello"


# ---------------------------------------------------------------------------
# _split_into_chunks
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    # -- basic behaviour -------------------------------------------------------

    def test_single_short_sentence_is_one_chunk(self):
        assert _split_into_chunks("Hello world.", max_words=10) == ["Hello world."]

    def test_empty_input_returns_empty_list(self):
        assert _split_into_chunks("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert _split_into_chunks("   \n\n  ") == []

    # -- sentence-boundary splitting --------------------------------------------

    def test_two_short_sentences_stay_together(self):
        text = "First. Second."
        chunks = _split_into_chunks(text, max_words=10)
        assert len(chunks) == 1

    def test_splits_at_sentence_boundary(self):
        # 6 words per sentence, limit 8 → flush after first sentence
        text = "One two three four five six. Seven eight nine ten eleven twelve."
        chunks = _split_into_chunks(text, max_words=8)
        assert len(chunks) == 2
        assert chunks[0].startswith("One")
        assert chunks[1].startswith("Seven")

    def test_multiple_sentence_boundaries(self):
        # 3 sentences × 4 words, limit 5 → one sentence per chunk
        text = "Aaa bbb ccc ddd. Eee fff ggg hhh. Iii jjj kkk lll."
        chunks = _split_into_chunks(text, max_words=5)
        assert len(chunks) == 3

    # -- hard-wrap on oversized sentences ---------------------------------------

    def test_single_oversized_sentence_is_hard_wrapped(self):
        words = " ".join(f"w{i}" for i in range(20))
        chunks = _split_into_chunks(words, max_words=5)
        assert len(chunks) == 4
        for chunk in chunks:
            assert len(chunk.split()) <= 5

    def test_oversized_sentence_after_short_flushes_buffer(self):
        short = "Short sentence."                            # 2 words
        long = " ".join(f"x{i}" for i in range(12))        # 12 words, no period
        text = f"{short} {long}"
        chunks = _split_into_chunks(text, max_words=5)
        # "Short sentence." flushed, then 12 words → 3 chunks of ≤5
        assert chunks[0] == "Short sentence."
        assert len(chunks) == 4  # 1 (short) + 3 (hard-wrapped)

    # -- edge cases -------------------------------------------------------------

    def test_single_word(self):
        assert _split_into_chunks("Hello", max_words=10) == ["Hello"]

    def test_exact_limit_does_not_create_extra_chunk(self):
        text = "a b c d e."          # exactly 5 words (with punctuation attached)
        chunks = _split_into_chunks(text, max_words=5)
        assert len(chunks) == 1

    def test_no_trailing_empty_chunks(self):
        text = "End sentence.   "
        chunks = _split_into_chunks(text, max_words=10)
        assert all(c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# _strip_markdown
# ---------------------------------------------------------------------------


class TestStripMarkdown:
    # -- headers ------------------------------------------------------------

    def test_h1(self):
        assert _strip_markdown("# Title") == "Title"

    def test_h3(self):
        assert _strip_markdown("### Sub") == "Sub"

    def test_h6(self):
        assert _strip_markdown("###### Deep") == "Deep"

    def test_header_mid_document(self):
        text = "Intro.\n\n## Section\n\nBody."
        result = _strip_markdown(text)
        assert "##" not in result
        assert "Section" in result

    # -- emphasis -----------------------------------------------------------

    def test_bold_asterisks(self):
        assert _strip_markdown("This is **bold** text.") == "This is bold text."

    def test_bold_underscores(self):
        assert _strip_markdown("This is __bold__ text.") == "This is bold text."

    def test_italic_asterisks(self):
        assert _strip_markdown("This is *italic* text.") == "This is italic text."

    def test_italic_underscores(self):
        assert _strip_markdown("This is _italic_ text.") == "This is italic text."

    def test_bold_inside_sentence(self):
        result = _strip_markdown("He said **hello** loudly.")
        assert result == "He said hello loudly."

    # -- links & images -----------------------------------------------------

    def test_link(self):
        assert _strip_markdown("[click here](https://example.com)") == "click here"

    def test_link_preserves_surrounding(self):
        result = _strip_markdown("See [docs](http://x.com) for info.")
        assert result == "See docs for info."

    def test_image_becomes_alt_text(self):
        assert _strip_markdown("![a photo](img.png)") == "a photo"

    def test_image_empty_alt(self):
        assert _strip_markdown("![](img.png)") == ""

    # -- code ---------------------------------------------------------------

    def test_inline_code(self):
        assert _strip_markdown("Use `pip install` now.") == "Use pip install now."

    def test_fenced_code_block_keeps_content(self):
        md = "Before.\n\n```python\nprint('hi')\n```\n\nAfter."
        result = _strip_markdown(md)
        assert "```" not in result
        assert "print('hi')" in result
        assert "Before." in result
        assert "After." in result

    # -- lists --------------------------------------------------------------

    def test_unordered_list_dash(self):
        result = _strip_markdown("- item one\n- item two")
        assert result == "item one\nitem two"

    def test_unordered_list_asterisk(self):
        result = _strip_markdown("* alpha\n* beta")
        assert result == "alpha\nbeta"

    def test_ordered_list(self):
        result = _strip_markdown("1. first\n2. second\n10. tenth")
        assert result == "first\nsecond\ntenth"

    # -- blockquote & hr ----------------------------------------------------

    def test_blockquote(self):
        result = _strip_markdown("> A wise quote.\n> Second line.")
        assert result == "A wise quote.\nSecond line."

    def test_horizontal_rule_dashes(self):
        result = _strip_markdown("Above.\n\n---\n\nBelow.")
        assert "---" not in result
        assert "Above." in result
        assert "Below." in result

    def test_horizontal_rule_asterisks(self):
        result = _strip_markdown("A.\n\n***\n\nB.")
        assert "***" not in result

    # -- strikethrough & html -----------------------------------------------

    def test_strikethrough(self):
        assert _strip_markdown("This is ~~deleted~~ text.") == "This is deleted text."

    def test_html_tag_removed(self):
        assert _strip_markdown("Hello <b>world</b>.") == "Hello world."

    def test_self_closing_html_tag(self):
        assert _strip_markdown("Line one.<br/>Line two.") == "Line one.Line two."

    # -- combined & edge cases ----------------------------------------------

    def test_plain_text_unchanged(self):
        plain = "Just a plain sentence with no formatting."
        assert _strip_markdown(plain) == plain

    def test_empty_string(self):
        assert _strip_markdown("") == ""

    def test_complex_document(self):
        md = (
            "# Welcome\n\n"
            "This is **bold** and *italic*.\n\n"
            "- Item [one](http://x)\n"
            "- Item `two`\n\n"
            "> A quote.\n\n"
            "---\n\n"
            "The end."
        )
        result = _strip_markdown(md)
        # No Markdown syntax should survive
        assert "#" not in result
        assert "**" not in result
        assert "*italic*" not in result
        assert "[" not in result
        assert "]" not in result
        assert "`" not in result
        assert ">" not in result.split("\n")[0] or result.split("\n")[0].strip() == ""
        assert "---" not in result
        # But the actual words should all be there
        for word in ("Welcome", "bold", "italic", "Item", "one", "two", "quote", "end"):
            assert word in result
