from datetime import date

from clinical_platform.security.phi_redaction import detokenize, irreversible_hash, shift_date, tokenize


def test_hash_deterministic():
    assert irreversible_hash("abc", "salt") == irreversible_hash("abc", "salt")


def test_tokenize_roundtrip():
    key = "k"
    t = tokenize("value", key)
    assert detokenize(t, key) == "value"
    assert detokenize(t, "wrong") is None


def test_shift_date():
    assert shift_date(date(2020, 1, 1), 10) == date(2020, 1, 11)

