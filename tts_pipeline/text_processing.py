"""Shared text processing utilities for TTS engines.

All text normalization, cleaning, and chunking functions used across
different TTS backends.
"""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Light normalization: strip BOM, collapse whitespace, trim.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text with BOM removed, whitespace collapsed, and trimmed
    """
    text = text.lstrip("\ufeff")  # BOM
    text = re.sub(r"\s+", " ", text)  # collapse
    return text.strip()


def strip_markdown(text: str) -> str:
    """Remove common Markdown syntax, returning plain text.

    Stripped constructs (in processing order):
        * Fenced code blocks  — content kept, fences removed
        * Inline code
        * Images              — alt text kept
        * Links               — link text kept
        * Strikethrough
        * Bold / italic
        * ATX headers (``#`` … ``######``)
        * Horizontal rules
        * Blockquotes
        * Unordered and ordered list markers
        * HTML tags

    Args:
        text: Markdown-formatted text

    Returns:
        Plain text with Markdown syntax removed
    """
    # Block-level (multi-line, must come first)
    text = re.sub(r"```[^\n]*\n(.*?)\n```", r"\1", text, flags=re.DOTALL)

    # Inline
    text = re.sub(r"`([^`]+)`", r"\1", text)  # code
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)  # image
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # link
    text = re.sub(r"~~(.+?)~~", r"\1", text)  # strikethrough
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # bold **
    text = re.sub(r"__(.+?)__", r"\1", text)  # bold __
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)  # italic *
    text = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"\1", text)  # italic _

    # Line-level
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headers
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)  # hr
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)  # blockquote
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)  # ul marker
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)  # ol marker

    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    return text


def split_into_chunks(text: str, max_words: int = 150) -> list[str]:
    """Sentence-aware text chunking for long documents.

    Splits on sentence-ending punctuation (.!?) followed by whitespace.
    If a single sentence exceeds max_words it is hard-wrapped on word
    boundaries so nothing is silently dropped.

    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk (default: 150)

    Returns:
        List of text chunks, each within the word limit
    """
    text = text.strip()
    if not text:
        return []

    # Split into sentences; regex keeps the punctuation with the sentence
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    buffer: list[str] = []  # sentences accumulated for the current chunk
    buf_words: int = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # Flush buffer if appending this sentence would bust the limit
        if buf_words + word_count > max_words and buffer:
            chunks.append(" ".join(buffer))
            buffer = []
            buf_words = 0

        if word_count > max_words:
            # Flush any remaining buffer first
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buf_words = 0
            # Hard-wrap the oversized sentence
            words = sentence.split()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
        else:
            buffer.append(sentence)
            buf_words += word_count

    if buffer:
        chunks.append(" ".join(buffer))

    return [c for c in chunks if c.strip()]


def split_by_char_limit(text: str, max_chars: int = 4096) -> list[str]:
    """Character-based text chunking for API limits.

    Splits on sentence boundaries when possible, falling back to word
    boundaries if individual sentences exceed the limit.

    Args:
        text: Input text to split
        max_chars: Maximum characters per chunk (default: 4096)

    Returns:
        List of text chunks, each within the character limit
    """
    if len(text) <= max_chars:
        return [text]

    # Try to split on sentences first
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    buffer: list[str] = []
    buf_len: int = 0

    for sentence in sentences:
        sent_len = len(sentence)

        # Flush buffer if adding this sentence would exceed limit
        if buf_len + sent_len + len(buffer) > max_chars and buffer:
            chunks.append(" ".join(buffer))
            buffer = []
            buf_len = 0

        # If single sentence exceeds limit, split on words
        if sent_len > max_chars:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buf_len = 0

            words = sentence.split()
            word_buffer: list[str] = []
            word_buf_len: int = 0

            for word in words:
                word_len = len(word)
                if word_buf_len + word_len + len(word_buffer) > max_chars:
                    if word_buffer:
                        chunks.append(" ".join(word_buffer))
                        word_buffer = []
                        word_buf_len = 0
                word_buffer.append(word)
                word_buf_len += word_len

            if word_buffer:
                chunks.append(" ".join(word_buffer))
        else:
            buffer.append(sentence)
            buf_len += sent_len

    if buffer:
        chunks.append(" ".join(buffer))

    return [c for c in chunks if c.strip()]
