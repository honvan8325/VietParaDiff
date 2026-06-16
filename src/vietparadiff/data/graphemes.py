"""Vietnamese grapheme decomposition and tokenization.

The tokenizer is deliberately compositional.  A syllable letter such as ``ắ`` is
represented as base ``a`` + structural modifier ``breve`` + tone ``acute``.  The
full composed glyph ID is still kept for CTC and evaluation, but generation
conditioning uses the decomposed streams so rare tone/modifier combinations can
share parameters with frequent ones.
"""

from __future__ import annotations

from dataclasses import dataclass
import string
import unicodedata

PAD = "<pad>"
UNK = "<unk>"
SPACE = "<space>"
NEWLINE = "<newline>"
NONE = "none"

TONE_MARKS = {
    "\u0301": "acute",
    "\u0300": "grave",
    "\u0309": "hook",
    "\u0303": "tilde",
    "\u0323": "dot",
}
MODIFIER_MARKS = {
    "\u0306": "breve",
    "\u0302": "circumflex",
    "\u031b": "horn",
}

BASE_CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
BASE_VOCAB = [PAD, UNK, SPACE, NEWLINE] + BASE_CHARS
MODIFIER_VOCAB = [PAD, NONE, "breve", "circumflex", "horn", "stroke"]
TONE_VOCAB = [PAD, NONE, "acute", "grave", "hook", "tilde", "dot"]
CASE_VOCAB = [PAD, NONE, "lower", "upper", "title", "other"]
TYPE_VOCAB = [PAD, "letter", "digit", "punct", "space", "newline", "symbol"]

# Explicit Vietnamese alphabet plus ASCII punctuation/digits.  The tokenizer can
# still map unseen Unicode characters to UNK.
VIETNAMESE_LETTERS = list(
    "aàáảãạăằắẳẵặâầấẩẫậ"
    "eèéẻẽẹêềếểễệ"
    "iìíỉĩị"
    "oòóỏõọôồốổỗộơờớởỡợ"
    "uùúủũụưừứửữự"
    "yỳýỷỹỵ"
    "dđ"
    "AÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
    "EÈÉẺẼẸÊỀẾỂỄỆ"
    "IÌÍỈĨỊ"
    "OÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "UÙÚỦŨỤƯỪỨỬỮỰ"
    "YỲÝỶỸỴ"
    "DĐ"
)
FULL_GRAPHEME_VOCAB = [PAD, UNK, SPACE, NEWLINE] + sorted(
    set(VIETNAMESE_LETTERS + list(string.ascii_letters + string.digits + string.punctuation))
)


@dataclass(frozen=True)
class GraphemeParts:
    """A decomposed Unicode grapheme used by all downstream modules."""

    surface: str
    base: str
    modifier: str
    tone: str
    case: str
    kind: str


def split_graphemes(text: str) -> list[str]:
    """Split Unicode text into base-plus-combining-mark clusters.

    This handles Vietnamese precomposed and decomposed marks after NFD
    normalization.  It intentionally keeps spaces and newlines as individual
    clusters because layout and HTR need them.
    """

    text = unicodedata.normalize("NFD", text)
    clusters: list[str] = []
    current = ""
    for ch in text:
        if unicodedata.combining(ch) and current:
            current += ch
        else:
            if current:
                clusters.append(unicodedata.normalize("NFC", current))
            current = ch
    if current:
        clusters.append(unicodedata.normalize("NFC", current))
    return clusters


def decompose_grapheme(cluster: str) -> GraphemeParts:
    """Decompose one cluster into Vietnamese compositional attributes."""

    if cluster == " ":
        return GraphemeParts(cluster, SPACE, NONE, NONE, NONE, "space")
    if cluster == "\n":
        return GraphemeParts(cluster, NEWLINE, NONE, NONE, NONE, "newline")
    if cluster == "":
        return GraphemeParts(cluster, UNK, NONE, NONE, NONE, "symbol")

    nfd = unicodedata.normalize("NFD", cluster)
    base = nfd[0]
    marks = nfd[1:]
    modifier = NONE
    tone = NONE
    if base in {"đ", "Đ"}:
        base = "d" if base == "đ" else "D"
        modifier = "stroke"
    for mark in marks:
        if mark in MODIFIER_MARKS:
            modifier = MODIFIER_MARKS[mark]
        elif mark in TONE_MARKS:
            tone = TONE_MARKS[mark]

    if cluster.isalpha():
        kind = "letter"
    elif cluster.isdigit():
        kind = "digit"
    elif cluster in string.punctuation or unicodedata.category(cluster[0]).startswith("P"):
        kind = "punct"
    else:
        kind = "symbol"

    if cluster[0].isupper():
        case = "upper"
    elif cluster[0].islower():
        case = "lower"
    elif cluster[0].istitle():
        case = "title"
    else:
        case = "other"
    return GraphemeParts(cluster, base, modifier, tone, case, kind)


class VietnameseTokenizer:
    """Aligned multi-stream tokenizer for Vietnamese paragraph generation."""

    def __init__(self, max_tokens: int | None = None) -> None:
        self.max_tokens = max_tokens
        self.vocabs = {
            "base": BASE_VOCAB,
            "modifier": MODIFIER_VOCAB,
            "tone": TONE_VOCAB,
            "case": CASE_VOCAB,
            "type": TYPE_VOCAB,
            "full": FULL_GRAPHEME_VOCAB,
        }
        self.stoi = {name: {tok: i for i, tok in enumerate(vocab)} for name, vocab in self.vocabs.items()}
        self.itos = {name: {i: tok for i, tok in enumerate(vocab)} for name, vocab in self.vocabs.items()}
        self.vocab_sizes = {name: len(vocab) for name, vocab in self.vocabs.items()}

    def parts(self, text: str) -> list[GraphemeParts]:
        parts = [decompose_grapheme(g) for g in split_graphemes(text)]
        return parts[: self.max_tokens] if self.max_tokens else parts

    def encode(self, text: str) -> dict[str, list[int] | list[GraphemeParts]]:
        parts = self.parts(text)
        encoded: dict[str, list[int] | list[GraphemeParts]] = {"parts": parts}
        for name in ["base", "modifier", "tone", "case", "type", "full"]:
            encoded[name] = []
        for part in parts:
            encoded["base"].append(self.stoi["base"].get(part.base, self.stoi["base"][UNK]))
            encoded["modifier"].append(self.stoi["modifier"].get(part.modifier, self.stoi["modifier"][NONE]))
            encoded["tone"].append(self.stoi["tone"].get(part.tone, self.stoi["tone"][NONE]))
            encoded["case"].append(self.stoi["case"].get(part.case, self.stoi["case"]["other"]))
            encoded["type"].append(self.stoi["type"].get(part.kind, self.stoi["type"]["symbol"]))
            surf = SPACE if part.surface == " " else NEWLINE if part.surface == "\n" else part.surface
            encoded["full"].append(self.stoi["full"].get(surf, self.stoi["full"][UNK]))
        return encoded

    def decode_full(self, ids: list[int]) -> str:
        out: list[str] = []
        for idx in ids:
            tok = self.itos["full"].get(int(idx), UNK)
            if tok == PAD:
                continue
            out.append(" " if tok == SPACE else "\n" if tok == NEWLINE else tok)
        return "".join(out)
