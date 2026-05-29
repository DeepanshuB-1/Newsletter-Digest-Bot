"""
tests/test_pipeline.py

Unit tests for pure functions across the pipeline.
No Ollama, no PostgreSQL, no network required.

Run with:  pytest tests/ -v
"""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
# classification.py
# ══════════════════════════════════════════════════════════════════════════════

from classification import _extract_address, _clean_body as _clean_body_cls


class TestExtractAddress:
    def test_name_angle_format(self):
        assert _extract_address('"The Rundown AI" <newsletter@therundown.ai>') == "newsletter@therundown.ai"

    def test_plain_email(self):
        assert _extract_address("hello@example.com") == "hello@example.com"

    def test_no_name(self):
        assert _extract_address("<user@gmail.com>") == "user@gmail.com"

    def test_lowercases_result(self):
        assert _extract_address("NEWS@EXAMPLE.COM") == "news@example.com"

    def test_strips_whitespace(self):
        assert _extract_address("  user@test.com  ") == "user@test.com"


class TestCleanBodyClassification:
    def test_removes_html_tags(self):
        html = "<div><p>Hello world</p></div>"
        result = _clean_body_cls(html)
        assert "<div>" not in result
        assert "Hello world" in result

    def test_removes_urls(self):
        text = "Check this out https://example.com/track?id=123 for more"
        result = _clean_body_cls(text)
        assert "https://" not in result
        assert "Check this out" in result

    def test_respects_max_chars(self):
        text = "a" * 1000
        result = _clean_body_cls(text, max_chars=100)
        assert len(result) <= 100

    def test_empty_input(self):
        assert _clean_body_cls("") == ""

    def test_collapses_whitespace(self):
        text = "hello    world\n\n\nfoo"
        result = _clean_body_cls(text)
        assert "  " not in result


# ══════════════════════════════════════════════════════════════════════════════
# gmail_oauth.py
# ══════════════════════════════════════════════════════════════════════════════

from gmail_oauth import _body_hash, _decode_mime_header, _html_to_text, _extract_links


class TestBodyHash:
    def test_returns_32_char_hex(self):
        h = _body_hash("some email body content")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_content_same_hash(self):
        assert _body_hash("hello world") == _body_hash("hello world")

    def test_different_content_different_hash(self):
        assert _body_hash("hello world") != _body_hash("hello worlds")

    def test_normalises_whitespace(self):
        # Extra spaces and case should not change the hash
        assert _body_hash("hello   world") == _body_hash("hello world")
        assert _body_hash("Hello World") == _body_hash("hello world")

    def test_uses_only_first_600_chars(self):
        base   = "x" * 600
        extra  = base + "different"
        assert _body_hash(base) == _body_hash(extra)

    def test_empty_string(self):
        h = _body_hash("")
        assert len(h) == 32


class TestDecodeMimeHeader:
    def test_plain_ascii(self):
        assert _decode_mime_header("Hello World") == "Hello World"

    def test_utf8_base64_encoded(self):
        # "AI News" in UTF-8 Base64 MIME encoding
        encoded = "=?UTF-8?B?QUkgTmV3cw==?="
        assert _decode_mime_header(encoded) == "AI News"

    def test_empty_string(self):
        assert _decode_mime_header("") == ""

    def test_none_fallback(self):
        # Should not raise, returns empty string
        assert _decode_mime_header(None) == ""


class TestHtmlToText:
    def test_strips_tags(self):
        html = "<h1>Title</h1><p>Body text here.</p>"
        result = _html_to_text(html)
        assert "<h1>" not in result
        assert "Title" in result
        assert "Body text here." in result

    def test_removes_script(self):
        html = "<script>alert('xss')</script><p>Content</p>"
        result = _html_to_text(html)
        assert "alert" not in result
        assert "Content" in result

    def test_removes_style(self):
        html = "<style>.foo{color:red}</style><p>Visible</p>"
        result = _html_to_text(html)
        assert "color" not in result
        assert "Visible" in result


class TestExtractLinks:
    def test_returns_list(self):
        html = '<a href="https://example.com/article">Read more details here</a>'
        links = _extract_links(html)
        assert isinstance(links, list)

    def test_extracts_url(self):
        html = '<a href="https://techcrunch.com/story/ai-breakthrough">Major AI Breakthrough Announced</a>'
        links = _extract_links(html)
        assert any("techcrunch.com" in l["url"] for l in links)

    def test_skips_social_media(self):
        html = '<a href="https://twitter.com/user">Twitter</a><a href="https://linkedin.com/post">LinkedIn</a>'
        links = _extract_links(html)
        assert len(links) == 0

    def test_skips_unsubscribe_links(self):
        html = '<a href="https://example.com/unsubscribe?id=123">Unsubscribe</a>'
        links = _extract_links(html)
        assert len(links) == 0

    def test_caps_at_five_links(self):
        html = "".join(
            f'<a href="https://news{i}.com/article">Article headline number {i} is interesting</a>'
            for i in range(10)
        )
        links = _extract_links(html)
        assert len(links) <= 5

    def test_empty_html(self):
        assert _extract_links("") == []
        assert _extract_links(None) == []


# ══════════════════════════════════════════════════════════════════════════════
# cleaner.py
# ══════════════════════════════════════════════════════════════════════════════

from cleaner import clean_body, _is_boilerplate_line


class TestIsBoilerplateLine:
    @pytest.mark.parametrize("line", [
        "Unsubscribe from this list",
        "View this email in your browser",
        "Privacy Policy | Terms of Service",
        "All rights reserved © 2026",
        "You are receiving this because you signed up",
        "Manage your email preferences",
        "Click here to opt out",
    ])
    def test_known_boilerplate(self, line):
        assert _is_boilerplate_line(line) is True

    @pytest.mark.parametrize("line", [
        "OpenAI releases GPT-5 with major reasoning improvements.",
        "The Federal Reserve held interest rates steady at 5.25%.",
        "SpaceX successfully launched the Starship rocket today.",
    ])
    def test_real_content(self, line):
        assert _is_boilerplate_line(line) is False

    def test_short_line_is_boilerplate(self):
        assert _is_boilerplate_line("hi") is True
        assert _is_boilerplate_line("OK") is True

    def test_boundary_length(self):
        # Exactly 15 chars — treated as boilerplate (< 15 check is strict)
        assert _is_boilerplate_line("a" * 14) is True
        assert _is_boilerplate_line("a" * 15) is False


class TestCleanBody:
    def test_removes_unsubscribe_lines(self):
        text = "Great article about AI.\nUnsubscribe from this list\nMore content here."
        result = clean_body(text)
        assert "unsubscribe" not in result.lower()
        assert "Great article" in result

    def test_removes_bare_urls(self):
        text = "Here is a story.\nhttps://track.example.com/pixel.gif\nAnd more text."
        result = clean_body(text)
        assert "track.example.com" not in result
        assert "Here is a story" in result

    def test_collapses_many_blank_lines(self):
        text = "Para one.\n\n\n\n\nPara two."
        result = clean_body(text)
        assert "\n\n\n" not in result

    def test_empty_input(self):
        assert clean_body("") == ""

    def test_preserves_real_content(self):
        content = "OpenAI announces new model. Google releases Gemini update. Meta open-sources Llama."
        assert clean_body(content) == content


# ══════════════════════════════════════════════════════════════════════════════
# summarizer.py — pure math / text functions only
# ══════════════════════════════════════════════════════════════════════════════

from summarizer import _chunk_body, _cosine, _centroid, _sender_name, _deduplicate


class TestChunkBody:
    def test_single_short_body(self):
        body = "This is a short paragraph."
        chunks = _chunk_body(body)
        assert len(chunks) >= 1

    def test_splits_on_double_newline(self):
        body = ("Paragraph one with enough content to matter.\n\n"
                "Paragraph two with enough content to matter.\n\n"
                "Paragraph three with enough content to matter.")
        # min_len=20 so 44-char paragraphs aren't filtered; chunk_size=50 forces splitting
        chunks = _chunk_body(body, chunk_size=50, min_len=20)
        assert len(chunks) >= 2

    def test_respects_chunk_size(self):
        long_para = "word " * 200          # ~1000 chars
        body = long_para + "\n\n" + long_para
        chunks = _chunk_body(body, chunk_size=700)
        assert all(len(c) <= 750 for c in chunks)  # small tolerance for boundary

    def test_empty_body(self):
        assert _chunk_body("") == []

    def test_filters_short_paragraphs(self):
        body = "Short.\n\nThis paragraph is long enough to be included in the output chunks."
        chunks = _chunk_body(body, min_len=20)
        # "Short." is only 6 chars — should be filtered
        assert not any(c.strip() == "Short." for c in chunks)


class TestCosine:
    def test_identical_vectors(self):
        v = [1.0, 0.5, 0.3]
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_score_between_minus1_and_1(self):
        import random
        random.seed(42)
        a = [random.gauss(0, 1) for _ in range(768)]
        b = [random.gauss(0, 1) for _ in range(768)]
        score = _cosine(a, b)
        assert -1.0 <= score <= 1.0


class TestCentroid:
    def test_single_vector(self):
        v = [1.0, 2.0, 3.0]
        assert _centroid([v]) == v

    def test_two_vectors_average(self):
        a = [0.0, 0.0]
        b = [2.0, 4.0]
        c = _centroid([a, b])
        assert abs(c[0] - 1.0) < 1e-9
        assert abs(c[1] - 2.0) < 1e-9

    def test_preserves_dimension(self):
        vecs = [[float(i) for i in range(768)] for _ in range(5)]
        result = _centroid(vecs)
        assert len(result) == 768


class TestSenderName:
    def test_name_angle_format(self):
        assert _sender_name('"Ben\'s Bites" <ben@bensbites.co>') == "Ben's Bites"

    def test_plain_email_uses_local_part(self):
        assert _sender_name("newsletter@example.com") == "newsletter"

    def test_no_name_with_angle(self):
        result = _sender_name("<user@example.com>")
        assert result  # something is returned


class TestDeduplicate:
    def _make_email(self, text):
        return {"subject": text, "body": text, "clean_body": text}

    def test_keeps_unique_items(self):
        dim = 10
        e1 = self._make_email("Story about AI breakthroughs")
        e2 = self._make_email("Story about climate change")
        v1 = [1.0] + [0.0] * (dim - 1)
        v2 = [0.0, 1.0] + [0.0] * (dim - 2)
        kept, _ = _deduplicate([e1, e2], [v1, v2])
        assert len(kept) == 2

    def test_removes_near_duplicates(self):
        dim = 10
        e1 = self._make_email("Duplicate story A")
        e2 = self._make_email("Duplicate story B")
        v1 = [1.0] + [0.0] * (dim - 1)
        v2 = [1.0] + [0.0] * (dim - 1)   # identical → cosine = 1.0 → dedup triggers
        kept, _ = _deduplicate([e1, e2], [v1, v2])
        assert len(kept) == 1


# ══════════════════════════════════════════════════════════════════════════════
# auth.py
# ══════════════════════════════════════════════════════════════════════════════

from auth import hash_password, verify_password


class TestAuth:
    def test_hash_is_not_plaintext(self):
        h = hash_password("mysecretpassword")
        assert h != "mysecretpassword"

    def test_verify_correct_password(self):
        h = hash_password("correct_password")
        assert verify_password("correct_password", h) is True

    def test_reject_wrong_password(self):
        h = hash_password("correct_password")
        assert verify_password("wrong_password", h) is False

    def test_same_password_different_hashes(self):
        # bcrypt salts are random — same password produces different hashes
        h1 = hash_password("password123")
        h2 = hash_password("password123")
        assert h1 != h2
        # but both should verify
        assert verify_password("password123", h1)
        assert verify_password("password123", h2)

    def test_empty_password_hashes(self):
        h = hash_password("")
        assert verify_password("", h) is True
        assert verify_password("x", h) is False
