"""Tests for Vietnamese base-shape-tone grapheme factorization."""

from __future__ import annotations

import unicodedata

import pytest

from vietparadiff.models.grapheme import FactorizedGrapheme, VietnameseGraphemeFactorizer


TONE_FORMS = {
    "none": 0,
    "acute": 1,
    "grave": 2,
    "hook_above": 3,
    "tilde": 4,
    "dot_below": 5,
}

VIETNAMESE_VOWEL_ROWS = (
    ("a", "none", "aáàảãạ"),
    ("a", "breve", "ăắằẳẵặ"),
    ("a", "circumflex", "âấầẩẫậ"),
    ("e", "none", "eéèẻẽẹ"),
    ("e", "circumflex", "êếềểễệ"),
    ("i", "none", "iíìỉĩị"),
    ("o", "none", "oóòỏõọ"),
    ("o", "circumflex", "ôốồổỗộ"),
    ("o", "horn", "ơớờởỡợ"),
    ("u", "none", "uúùủũụ"),
    ("u", "horn", "ưứừửữự"),
    ("y", "none", "yýỳỷỹỵ"),
)

VIETNAMESE_VOWEL_CASES = tuple(
    (surface, base, shape, tone, "lower")
    for base, shape, forms in VIETNAMESE_VOWEL_ROWS
    for tone, index in TONE_FORMS.items()
    for surface in (forms[index],)
) + tuple(
    (surface.upper(), base, shape, tone, "upper")
    for base, shape, forms in VIETNAMESE_VOWEL_ROWS
    for tone, index in TONE_FORMS.items()
    for surface in (forms[index],)
)


@pytest.fixture
def factorizer() -> VietnameseGraphemeFactorizer:
    return VietnameseGraphemeFactorizer()


@pytest.mark.parametrize(
    ("surface", "base", "shape", "tone", "case"),
    VIETNAMESE_VOWEL_CASES,
)
def test_factorizes_every_vietnamese_vowel_form(
    factorizer: VietnameseGraphemeFactorizer,
    surface: str,
    base: str,
    shape: str,
    tone: str,
    case: str,
) -> None:
    assert factorizer.factorize(surface) == [
        FactorizedGrapheme(
            surface=surface,
            base=base,
            shape=shape,
            tone=tone,
            case=case,
            class_name="letter",
        )
    ]


@pytest.mark.parametrize(
    ("surface", "case"),
    (("đ", "lower"), ("Đ", "upper")),
)
def test_factorizes_d_with_bar(
    factorizer: VietnameseGraphemeFactorizer,
    surface: str,
    case: str,
) -> None:
    assert factorizer.factorize(surface) == [
        FactorizedGrapheme(
            surface=surface,
            base="d",
            shape="bar",
            tone="none",
            case=case,
            class_name="letter",
        )
    ]


@pytest.mark.parametrize("surface", ("ậ", "Ắ", "ớ", "Ự"))
def test_nfc_and_nfd_inputs_are_equivalent_and_output_nfc(
    factorizer: VietnameseGraphemeFactorizer,
    surface: str,
) -> None:
    from_nfc = factorizer.factorize(surface)
    from_nfd = factorizer.factorize(unicodedata.normalize("NFD", surface))

    assert from_nfd == from_nfc
    assert from_nfd[0].surface == unicodedata.normalize("NFC", surface)


def test_preserves_token_order_and_classifies_non_letters(
    factorizer: VietnameseGraphemeFactorizer,
) -> None:
    result = factorizer.factorize("A1 đ!\n\t\x00")

    assert result == [
        FactorizedGrapheme("A", "a", "none", "none", "upper", "letter"),
        FactorizedGrapheme("1", "1", "none", "none", "none", "digit"),
        FactorizedGrapheme(" ", " ", "none", "none", "none", "space"),
        FactorizedGrapheme("đ", "d", "bar", "none", "lower", "letter"),
        FactorizedGrapheme("!", "!", "none", "none", "none", "punctuation"),
        FactorizedGrapheme("\n", "\n", "none", "none", "none", "newline"),
        FactorizedGrapheme(" ", " ", "none", "none", "none", "space"),
        FactorizedGrapheme("\x00", "\x00", "none", "none", "none", "special"),
    ]


def test_normalizes_all_non_newline_whitespace_to_space(
    factorizer: VietnameseGraphemeFactorizer,
) -> None:
    result = factorizer.factorize(" \t\u00a0")

    assert [item.surface for item in result] == [" ", " ", " "]
    assert all(item.class_name == "space" for item in result)


def test_unknown_combining_mark_does_not_become_shape_or_tone(
    factorizer: VietnameseGraphemeFactorizer,
) -> None:
    result = factorizer.factorize("a\u0304")

    assert result == [
        FactorizedGrapheme(
            surface="ā",
            base="a",
            shape="none",
            tone="none",
            case="lower",
            class_name="letter",
        )
    ]


@pytest.mark.parametrize("invalid", (None, 1, b"abc", ["a"]))
def test_rejects_non_string_input(
    factorizer: VietnameseGraphemeFactorizer,
    invalid: object,
) -> None:
    with pytest.raises(TypeError, match="text phải là str"):
        factorizer.factorize(invalid)  # type: ignore[arg-type]


def test_rejects_empty_text(
    factorizer: VietnameseGraphemeFactorizer,
) -> None:
    with pytest.raises(ValueError, match="text không được rỗng"):
        factorizer.factorize("")


def test_rejects_leading_combining_mark(
    factorizer: VietnameseGraphemeFactorizer,
) -> None:
    with pytest.raises(
        ValueError,
        match="Combining mark không có ký tự cơ sở",
    ):
        factorizer.factorize("\u0301a")
