"""
Text normalization for TTS preprocessing (from longecho, https://github.com/suraj-geddam/longecho, MIT).

Converts text patterns that TTS models struggle with into speakable forms:
- "$5M" → "5 million dollars"
- "Dr. Smith" → "Doctor Smith"
- "(important note)" → "important note" (removes parens, keeps content)

Usage:
    result = TextNormalizer().normalize("The cost is $5M.")

Plain numbers are left for the TTS model, which reads them in context ("in 1999" vs "1999 people").
spell_numbers() is used by the take checker, to compare a script with Whisper's transcript.
"""

import logging
import re

logger = logging.getLogger(__name__)


def spell_numbers(s: str) -> str:
    """Numbers to words, any size: "1,250" -> "one thousand, two hundred and fifty", "3.5" -> "three point
    five", "21st" -> "twenty-first", "1843" -> "eighteen forty-three" (four-digit numbers 1000-2099 without a
    comma read as years)."""
    from decimal import Decimal
    from num2words import num2words

    def rep(m):
        t = m.group(0).replace(",", "")
        try:
            if re.fullmatch(r"(1[0-9]|20)[0-9]{2}", m.group(0)):  # "1843" is a year, "1,843" isn't
                return num2words(int(t), to="year")
            if t.endswith(("st", "nd", "rd", "th")):
                return num2words(int(t[:-2]), to="ordinal")
            return num2words(Decimal(t) if "." in t else int(t))
        except Exception:
            return t
    return re.sub(r"\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th)?\b", rep, s)

class TextNormalizer:
    """Text normalizer for TTS preprocessing: currency, abbreviations, symbols, parentheses."""

    def normalize(self, text: str) -> str:
        """
        Normalize text for TTS processing.

        Pipeline:
        1. Expand currencies, abbreviations, symbols
        2. Strip parentheses but keep content inside

        Args:
            text: Input text to normalize

        Returns:
            Normalized text ready for TTS
        """
        if not text or not text.strip():
            return text

        try:
            # Step 1: Apply normalization
            normalized = self._apply_normalization(text)

            # Step 2: Strip parentheses but keep content
            normalized = self._strip_parentheses(normalized)

            return normalized

        except Exception as e:
            logger.warning(f"Text normalization failed, returning original: {e}")
            return text

    def _apply_normalization(self, text: str) -> str:
        """
        Apply regex-based normalization.

        Handles common patterns:
        - Currency: $5, $5M, $5B, $5K
        - Common abbreviations: Dr., Mr., Mrs., Ms., etc.
        """
        result = text

        # Currency with multipliers: $5M, $5B, $5K, $5T
        result = re.sub(
            r'\$(\d+(?:\.\d+)?)\s*([KkMmBbTt])\b',
            self._expand_currency_multiplier,
            result
        )

        # Currency used as an adjective: "an $8 flat" -> "an 8-dollar flat"
        result = re.sub(r'\b([Aa]n?) \$(\d+)\b(?= [a-z])', r'\1 \2-dollar', result)

        # Simple currency: $5, $123.45, $1,000
        result = re.sub(
            r'\$(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\b',
            self._expand_simple_currency,
            result
        )

        # Common abbreviations
        abbreviations = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\bMme\.': 'Madame',
            r'\bProf\.': 'Professor',
            r'\bSt\.': 'Saint',
            r'\bJr\.': 'Junior',
            r'\bSr\.': 'Senior',
            r'\bvs\.': 'versus',
            r'\betc\.': 'etcetera',
            r'\be\.g\.': 'for example',
            r'\bi\.e\.': 'that is',
        }
        for pattern, replacement in abbreviations.items():
            result = re.sub(pattern, replacement, result)

        # Symbol expansions for TTS
        # Order matters - do these before number conversion
        result = re.sub(r'\s*=\s*', ' equals ', result)  # = to equals
        result = re.sub(r'\s*\+\s*', ' plus ', result)   # + to plus
        result = re.sub(r'&', ' and ', result)           # & to and
        result = re.sub(r'(\d+)\s*%', r'\1 percent', result)  # 50% to 50 percent
        result = re.sub(r'\s*@\s*', ' at ', result)      # @ to at
        # Only expand slash between pure alphabetic words, but NOT in URLs/paths
        # Require space or start-of-string before the first word to avoid URL segments
        result = re.sub(r'(?<![:/.\w])([a-zA-Z]+)/([a-zA-Z]+)(?![:/.\w])', r'\1 or \2', result)

        # Clean up extra spaces from symbol expansion
        result = re.sub(r' +', ' ', result)

        return result

    def _expand_currency_multiplier(self, match: re.Match) -> str:
        """Expand currency with multiplier like $5M to 5 million dollars."""
        amount = match.group(1)
        multiplier = match.group(2).upper()

        multiplier_words = {
            'K': 'thousand',
            'M': 'million',
            'B': 'billion',
            'T': 'trillion',
        }

        word = multiplier_words.get(multiplier, '')
        return f"{amount} {word} dollars"

    def _expand_simple_currency(self, match: re.Match) -> str:
        """Expand simple currency: $5 -> 5 dollars, $1.87 -> 1 dollar and 87 cents, $0.50 -> 50 cents."""
        dollars, _, cents = match.group(1).partition('.')
        unit = lambda n, word: f"{n} {word}" + ("" if n == "1" else "s")
        cents = str(int((cents + "0")[:2])) if cents else ""  # .5 -> 50, .05 -> 5
        if not cents or cents == "0":
            return unit(dollars, "dollar")
        if dollars.replace(",", "") == "0":
            return unit(cents, "cent")
        return f"{unit(dollars, 'dollar')} and {unit(cents, 'cent')}"

    def _strip_parentheses(self, text: str) -> str:
        """
        Remove parentheses but keep the content inside, adding commas for speech pauses.

        "(important note)" → "important note"
        "Hello (world) there" → "Hello, world, there"

        WhisperD uses parens for non-speech events like "(coughs)",
        so we need to strip them to avoid content being skipped.
        Adding commas creates natural pauses where parentheses were.
        """
        # Replace parenthetical content with comma-delimited version
        # Pattern: text before ( content ) text after
        # We want: text before, content, text after

        def add_commas(match: re.Match) -> str:
            before = match.group(1)  # Text before opening paren
            content = match.group(2)  # Content inside parens
            after = match.group(3)    # Text after closing paren

            # Add comma before content if there's text before and no punctuation already
            if before and before.rstrip() and not before.rstrip()[-1] in ',:;-':
                before = before.rstrip() + ','

            # Add comma after content if there's text after (not just punctuation)
            after_stripped = after.lstrip()
            if after_stripped and after_stripped[0].isalnum():
                content = content + ','

            return f"{before} {content} {after}"

        # Process parentheses one at a time (handles nested by iterating)
        result = text
        while '(' in result and ')' in result:
            # Match: (optional text before)(content)(optional text after)
            new_result = re.sub(
                r'(^|.*?)(?<![(\s])\s*\(([^()]+)\)\s*(?![)\s])(.*)$',
                add_commas,
                result,
                count=1
            )
            if new_result == result:
                # Fallback: just strip remaining parens
                result = result.replace('(', '').replace(')', '')
                break
            result = new_result

        # Clean up any double spaces, double commas, or comma-space-comma
        result = re.sub(r',\s*,', ',', result)
        result = re.sub(r' +', ' ', result)
        # Remove space before punctuation
        result = re.sub(r' ([.,!?;:])', r'\1', result)

        return result.strip()
