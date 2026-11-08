"""Tests for text_normalizer module."""

import pytest


class TestStripParentheses:
    """Tests for parentheses stripping functionality."""

    def test_strip_simple_parentheses(self):
        """Should remove parentheses and keep content with comma."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("(important note)")
        # Standalone parens just become the content
        assert result == "important note"

    def test_strip_parentheses_in_sentence(self):
        """Should add comma before parenthetical content for natural pause."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("Hello (world) there")
        # Should add comma before and after for speech pause
        assert result == "Hello, world, there"

    def test_strip_multiple_parentheses(self):
        """Should handle multiple sets of parentheses with commas."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("(first) and (second)")
        # Each parenthetical gets comma treatment
        assert result == "first, and, second"

    def test_strip_nested_parentheses(self):
        """Should handle nested parentheses."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("((nested))")
        assert result == "nested"

    def test_strip_parentheses_no_parens(self):
        """Should return text unchanged if no parentheses."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("no parens here")
        assert result == "no parens here"

    def test_strip_parentheses_cleans_double_spaces(self):
        """Should clean up double spaces after removing parentheses."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("word (removed) word")
        # Should have commas instead of spaces around parenthetical
        assert "  " not in result
        assert result == "word, removed, word"

    def test_strip_parentheses_at_end_of_sentence(self):
        """Should handle parentheses at end of sentence."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("See details (below).")
        # Comma before, period stays at end
        assert result == "See details, below."

    def test_strip_parentheses_preserves_existing_punctuation(self):
        """Should not double up punctuation."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._strip_parentheses("Hello, (world) there")
        # Already has comma before, don't add another
        assert result == "Hello, world, there"
        assert ",," not in result


class TestNormalization:
    """Tests for text normalization."""

    def test_currency_with_multiplier_millions(self):
        """Should expand $5M to 5 million dollars."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("The cost is $5M.")
        assert "5 million dollars" in result
        assert "$" not in result

    def test_currency_with_multiplier_billions(self):
        """Should expand $2.5B to 2.5 billion dollars."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Revenue: $2.5B")
        assert "2.5 billion dollars" in result

    def test_currency_with_multiplier_thousands(self):
        """Should expand $100K to 100 thousand dollars."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Salary: $100K")
        assert "100 thousand dollars" in result

    def test_simple_currency(self):
        """Should expand $5 to 5 dollars."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("It costs $5.")
        assert "5 dollars" in result
        assert "$" not in result

    def test_currency_with_cents(self):
        """Should expand $5.99 to 5 dollars and 99 cents."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Price: $5.99")
        assert "5 dollars and 99 cents" in result

    def test_abbreviation_dr(self):
        """Should expand Dr. to Doctor."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Dr. Smith arrived.")
        assert "Doctor Smith" in result
        assert "Dr." not in result

    def test_abbreviation_mr_mrs(self):
        """Should expand Mr. and Mrs."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Mr. and Mrs. Jones")
        assert "Mister" in result
        assert "Missus" in result

    def test_abbreviation_etc(self):
        """Should expand etc."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("apples, oranges, etc.")
        assert "etcetera" in result

    def test_symbol_equals(self):
        """Should expand = to equals."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("experience = credibility")
        assert "experience equals credibility" in result
        assert " = " not in result

    def test_symbol_plus(self):
        """Should expand + to plus."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("2 + 2")
        assert "2 plus 2" in result

    def test_symbol_ampersand(self):
        """Should expand & to and."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("R&D department")
        assert "R and D" in result

    def test_symbol_percent(self):
        """Should expand % to percent."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("50% off")
        assert "50 percent" in result

    def test_symbol_at(self):
        """Should expand @ to at."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("email @ domain")
        assert "email at domain" in result

    def test_symbol_slash_as_or(self):
        """Should expand / to or in word contexts."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("yes/no question")
        assert "yes or no" in result

    def test_symbol_slash_preserves_urls(self):
        """Should NOT expand slashes in URLs."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Visit http://example.com/path")
        assert "http://example.com/path" in result
        assert " or " not in result

    def test_symbol_slash_preserves_dates(self):
        """Should NOT expand slashes in dates."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer._apply_normalization("Date: 12/25/2024")
        assert "12/25/2024" in result
        assert " or " not in result

    def test_preserves_plain_numbers(self):
        """Plain numbers are left for the TTS model."""
        from qwen3_longform.normalize import TextNormalizer
        result = TextNormalizer()._apply_normalization("I have 42 items.")
        assert "42" in result


class TestNormalizeEndToEnd:
    """End-to-end tests for the normalize() method."""

    def test_normalize_empty_string(self):
        """Empty string should return empty string."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("   ") == "   "

    def test_normalize_currency_and_parens(self):
        """Should handle currency and strip parentheses."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer.normalize("($5M)")
        assert "5 million dollars" in result
        assert "(" not in result
        assert ")" not in result

    def test_normalize_preserves_plain_text(self):
        """Plain text without special patterns should be preserved."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer.normalize("Hello world")
        assert result == "Hello world"

    def test_normalize_complex_sentence(self):
        """Should handle complex sentences."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        result = normalizer.normalize("Dr. Smith invested $5M (see details below).")
        assert "Doctor" in result
        assert "million dollars" in result
        assert "(" not in result
        assert ")" not in result

    def test_normalize_error_returns_original(self):
        """Should return original text if normalization fails."""
        from qwen3_longform.normalize import TextNormalizer
        normalizer = TextNormalizer()
        # Normal text should work fine
        result = normalizer.normalize("Simple text")
        assert result == "Simple text"


class TestCurrencyDetails:
    """Singular units, short cents and thousands separators."""

    @pytest.mark.parametrize("text,expected", [
        ("$1", "1 dollar"),
        ("$1.87", "1 dollar and 87 cents"),
        ("$1.5", "1 dollar and 50 cents"),
        ("$2.05", "2 dollars and 5 cents"),
        ("$2.01", "2 dollars and 1 cent"),
        ("$0.50", "50 cents"),
        ("$1,000", "1,000 dollars"),
        ("$20,", "20 dollars,"),
    ])
    def test_amounts(self, text, expected):
        from qwen3_longform.normalize import TextNormalizer
        assert TextNormalizer().normalize(text) == expected

    def test_script_and_transcript_compare_equal(self):
        from qwen3_longform.metrics import wer_ops
        assert wer_ops("She had only $1.87.", "She had only one dollar and eighty-seven cents.")[0] == 0
        assert wer_ops("She had only one dollar and eighty-seven cents.", "She had only $1.87.")[0] == 0


def test_currency_as_adjective():
    from qwen3_longform.normalize import TextNormalizer
    assert TextNormalizer().normalize("a pier glass in an $8 flat") == "a pier glass in an 8-dollar flat"
    assert TextNormalizer().normalize("a flat at $8 per week") == "a flat at 8 dollars per week"


def test_sentences_keep_initials():
    from qwen3_longform.script import sentences
    assert sentences("A modest D. But Young came home. Then J. R. Smith left.") == \
        ["A modest D. But Young came home.", "Then J. R. Smith left."]


def test_script_normalized_before_chunking(tmp_path):
    from qwen3_longform.script import load_script, make_chunks
    p = tmp_path / "s.txt"
    p.write_text("# One\n\nMr. Young had only $1.87. Dr. Smith lent him (kindly) an $8 coat.\n")
    [c] = make_chunks(load_script(str(p)))
    assert c["text"] == "Mister Young had only 1 dollar and 87 cents. Doctor Smith lent him, kindly, an 8-dollar coat."


@pytest.mark.parametrize("text,expected", [
    ("I have 42 items.", "I have forty-two items."),
    ("About 1,250 people came.", "About one thousand, two hundred and fifty people came."),
    ("It cost $999,999,999,999,999.99.",
     "It cost nine hundred and ninety-nine trillion, nine hundred and ninety-nine billion, nine hundred and "
     "ninety-nine million, nine hundred and ninety-nine thousand, nine hundred and ninety-nine dollars and "
     "ninety-nine cents."),
    ("It was the 21st of May, 1843.", "It was the twenty-first of May, eighteen forty-three."),
    ("Add 3.5 cups.", "Add three point five cups."),
])
def test_spell_numbers_any_size(text, expected):
    from qwen3_longform.normalize import TextNormalizer, spell_numbers
    assert spell_numbers(TextNormalizer().normalize(text)) == expected


def test_checker_ignores_spelling_variants():
    from qwen3_longform.metrics import wer_ops
    assert wer_ops("a grey cat in the harbour", "a gray cat in the harbor")[0] == 0
    assert wer_ops("keep 'em a while", "keep them a while")[0] == 0
    assert wer_ops("the sign read: Mme. Sofronie.", "the sign read Madame Sofronie")[0] == 0


def test_checker_rejoins_split_decimals():
    from qwen3_longform.metrics import wer_ops
    # Whisper's word timestamps give "$1" and ".87" as separate words
    assert wer_ops("she had only 1 dollar and 87 cents", "she had only $1 .87")[0] == 0
