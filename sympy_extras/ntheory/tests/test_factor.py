"""The Carmichael numbers of the typed ntheory against SymPy."""
from __future__ import annotations


def test_carmichael_numbers_match_sympy() -> None:
    from sympy.ntheory.factor_ import find_carmichael_numbers_in_range as sympy_range
    from sympy.ntheory.factor_ import find_first_n_carmichaels as sympy_first
    from sympy.ntheory.factor_ import is_carmichael as sympy_is
    from sympy_extras.ntheory import find_carmichael_numbers_in_range, find_first_n_carmichaels, is_carmichael
    assert all(is_carmichael(n) == bool(sympy_is(n)) for n in range(1, 20000))
    assert find_carmichael_numbers_in_range(0, 100000) == sympy_range(0, 100000)
    assert find_carmichael_numbers_in_range(561, 9000) == sympy_range(561, 9000)
    assert find_first_n_carmichaels(12) == sympy_first(12)
