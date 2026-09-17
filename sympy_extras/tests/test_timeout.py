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


def test_manualintegrate_patterns_are_complete_after_an_interrupted_build() -> None:
    # SymPy's special_function_rule builds its patterns on first use,
    # the wildcards first and the patterns after; a limit hit in between
    # left the wildcards in place and the patterns empty, the next call
    # doubled the wildcards, and exp(exp(x)) lost its Ei for the rest of
    # the process (two census entries after a slow one in the same
    # worker): the lists are rebuilt before a limit is set
    from sympy import Ei, exp, symbols
    from sympy.integrals.manualintegrate import manualintegrate
    from sympy.testing.pytest import raises
    import sympy.integrals.manualintegrate as manual
    x = symbols('x')
    assert attempt(lambda: 1, 5) == 1
    wilds = list(manual._wilds)
    assert len(wilds) == 5 and manual._special_function_patterns
    # the half-built state: the next call doubles the wildcards and the
    # Ei rule gets six arguments
    manual._special_function_patterns.clear()
    raises(TypeError, lambda: manualintegrate(exp(exp(x)), x))
    assert len(manual._wilds) == 10
    # and the error left the recursion marks of exp(exp(x)) and of the
    # exp(u)/u inside it in the cache of integral_steps: DontKnowRule for
    # them ever after
    assert manual._integral_cache and manualintegrate(exp(x) / x, x) != Ei(x)
    assert attempt(lambda: 1, 5) == 1
    assert len(manual._wilds) == 5 and manual._special_function_patterns and not manual._integral_cache
    assert manualintegrate(exp(x) * exp(-x + exp(x)), x) == Ei(exp(x))
    assert manualintegrate(exp(x) / x, x) == Ei(x)


def test_attempt_contains_a_polynomial_error() -> None:
    # sympy-extras: SymPy's singularities() raises PolynomialError from
    # CRootOf on a polynomial in two symbols, and it escaped attempt() as
    # a crash, so integrate_by_ranges(1, Or(x**2 + y**2 < 1,
    # (x - 1)**2 + y**2 < 1)) died instead of trying another route
    from sympy.polys.polyerrors import PolynomialError

    def raiser() -> int:
        raise PolynomialError("only univariate polynomials are allowed")

    assert attempt(raiser, 5) is None


def test_the_limit_survives_a_bare_except() -> None:
    # the bug: TimeLimitExceeded was an Exception, and SymPy's routines
    # catch Exception in places; once swallowed there the limit was gone
    # for the rest of the computation (integrals ran for hundreds of
    # seconds under a limit of twenty)
    import time

    def swallowing() -> int:
        started = time.monotonic()
        while time.monotonic() - started < 5:
            try:
                time.sleep(0.01)
            except Exception:
                pass
        return 1

    started = time.monotonic()
    assert attempt(swallowing, 0.2) is None
    assert time.monotonic() - started < 2
