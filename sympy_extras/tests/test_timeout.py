"""Tests of the time limit helper."""
from __future__ import annotations

from sympy_extras._timeout import attempt


def test_attempt_contains_a_recursion_error() -> None:
    # sympy-extras#49: attempt() is the package's way of containing a
    # SymPy call that gives up, and is used around exactly the calls that
    # can exhaust the stack, but it let RecursionError through -- one
    # equation deep in dsolve ended four benchmark sweeps out of four.
    def bottomless() -> int:
        return bottomless() + 1

    assert attempt(bottomless, 5) is None


def test_attempt_returns_the_value_otherwise() -> None:
    assert attempt(lambda: 42, 5) == 42


def test_sympy_tables_are_complete_after_an_interrupted_build() -> None:
    # SymPy builds its table of Meijer G representations on first use;
    # a limit hit during the build left the global truncated, and every
    # later integration of an exponential went astray (the Mellin
    # transform of exp(-x) came out as uppergamma(s, 0) on the whole plane
    # and a divergent integral got a value from it): the table is built
    # before a limit is set, and installed only when complete
    import sympy.integrals.meijerint as meijerint
    from sympy_extras import _timeout
    vars(meijerint)['_lookup_table'] = None
    _timeout._tables_complete = False
    assert attempt(lambda: 1, 5) == 1
    table = meijerint._lookup_table
    assert table is not None and sum(len(entries) for entries in table.values()) > 40
