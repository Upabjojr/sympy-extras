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


def test_the_limits_are_scaled_for_a_slower_machine() -> None:
    # the bug: the continuous integration, on machines about half as fast,
    # failed for a week on an integral which takes 9 s of its limit of 15 s
    # on the machine where the limits were chosen, and came out unevaluated
    # there; the examples of the documentation, which run after the tests,
    # went unchecked meanwhile. settings.time_scale, set from the
    # environment, multiplies every limit
    import time
    from sympy.testing.pytest import raises
    from sympy_extras.settings import _time_scale_of_the_environment, configure, settings

    def half_a_second() -> int:
        started = time.monotonic()
        while time.monotonic() - started < 0.5:
            pass
        return 1

    with configure(time_scale=1.0):
        assert attempt(half_a_second, 0.1) is None
    with configure(time_scale=50.0):
        assert attempt(half_a_second, 0.1) == 1
    assert settings.time_scale == _time_scale_of_the_environment()
    raises(ValueError, lambda: configure(time_scale=0.0).__enter__())
    import os
    previous = os.environ.get('SYMPY_EXTRAS_TIME_SCALE')
    try:
        for given, expected in [('3', 3.0), ('0.5', 0.5), ('', 1.0), ('fast', 1.0), ('-2', 1.0), ('inf', 1.0)]:
            os.environ['SYMPY_EXTRAS_TIME_SCALE'] = given
            assert _time_scale_of_the_environment() == expected
    finally:
        if previous is None:
            del os.environ['SYMPY_EXTRAS_TIME_SCALE']
        else:
            os.environ['SYMPY_EXTRAS_TIME_SCALE'] = previous


def _spin(seconds: float) -> int:
    """Busy for ``seconds`` (a computation which does not sleep)."""
    import time
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        pass
    return 1


def test_an_enclosing_limit_goes_to_its_owner() -> None:
    # the bug: an inner attempt() cut short by the enclosing limit took the
    # expiry for its own and returned None, and the caller went on without
    # the step (a census integral got a case valued as an unevaluated
    # Integral); the expiry goes to the owner of the enclosing limit
    import time
    went_on: list[bool] = []

    def outer() -> int:
        inner = attempt(lambda: _spin(5), 3)
        went_on.append(inner is None)
        return 1

    started = time.monotonic()
    assert attempt(outer, 0.3) is None
    assert went_on == []
    assert time.monotonic() - started < 2


def test_an_inner_limit_alone_gives_none() -> None:
    # an inner limit which expires before the enclosing one is the inner
    # step failing: None there, and the enclosing computation goes on
    assert attempt(lambda: (attempt(lambda: _spin(5), 0.1), 'went on'), 5) == (None, 'went on')


def test_nested_limits_each_take_their_own_expiry() -> None:
    # three limits: the middle one expires during the innermost, so the
    # innermost lets it through, the middle one gives None and the
    # outermost goes on
    went_on: list[str] = []

    def middle() -> int:
        attempt(lambda: _spin(5), 4)
        went_on.append('middle')
        return 1

    def outer() -> str:
        found = attempt(middle, 0.3)
        went_on.append('outer')
        return 'outer' if found is None else 'wrong'

    assert attempt(outer, 10) == 'outer'
    assert went_on == ['outer']


def test_an_attempt_without_a_limit_lets_the_enclosing_one_through() -> None:
    # the bug: attempt(f, None), which sets no limit of its own, still took
    # the expiry of the enclosing limit for f failing
    went_on: list[bool] = []

    def outer() -> int:
        went_on.append(attempt(lambda: _spin(5), None) is None)
        return 1

    assert attempt(outer, 0.3) is None
    assert went_on == []


def test_an_expiry_swallowed_on_the_way_is_raised_again() -> None:
    # the bug: code which swallows the expiry of the enclosing limit (as
    # SymPy's catch-all handlers would) went on with every later attempt()
    # failing at once, each taking the re-armed expiry for its own; a limit
    # entered after its enclosing one has expired raises that expiry
    from sympy_extras._timeout import TimeLimitExceeded
    went_on: list[bool] = []

    def swallowing() -> int:
        try:
            _spin(5)
        except TimeLimitExceeded:
            pass
        went_on.append(attempt(lambda: 1, 5) is None)
        return 1

    assert attempt(swallowing, 0.3) is None
    assert went_on == []


def test_the_limit_of_the_owner_is_restored() -> None:
    # after an inner limit, the enclosing one keeps its own deadline
    import time
    from sympy_extras._timeout import remaining_time, time_limit
    with time_limit(10):
        assert attempt(lambda: _spin(5), 0.1) is None
        left = remaining_time()
        assert left is not None and 9 < left * 1 <= 10
    assert remaining_time() is None
    started = time.monotonic()
    assert attempt(lambda: _spin(5), 0.2) is None
    assert time.monotonic() - started < 2


def test_limits_entered_as_the_enclosing_one_expires_leave_nothing_behind() -> None:
    # the bug: an expiry raised while a limit was being entered or left
    # (between pushing it on the stack of limits and arming the timer)
    # escaped its cleanup: the limit stayed on the stack, the timer was
    # armed again after the handler had been restored, and SIGALRM killed
    # the process (the documentation run died with exit code 142); before
    # the stack, the same race let the expiry out of the owner's attempt()
    import signal
    from sympy_extras import _timeout

    def churn() -> int:
        while True:
            attempt(lambda: attempt(lambda: 1, 5), 5)

    handler = signal.getsignal(signal.SIGALRM)
    for _ in range(100):
        assert attempt(churn, 0.01) is None
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
        assert signal.getsignal(signal.SIGALRM) == handler
        assert _timeout._running == []


def test_a_limit_shorter_than_its_entry_is_the_owners() -> None:
    # a limit which expired while being entered raised its expiry before
    # the owner had its Deadline: attempt could not tell it was its own
    # and let TimeLimitExceeded through, instead of returning None (the
    # Gröbner dispatcher's direct computation under 1e-6 s killed its
    # caller, sympy-extras CI of 2026-10-06)
    for seconds in (1e-9, 1e-7, 1e-6):
        for _ in range(20):
            assert attempt(lambda: sum(range(10**5)), seconds) in (None, sum(range(10**5)))
    # inside an enclosing limit with time left, the same
    assert attempt(lambda: attempt(lambda: sum(range(10**5)), 1e-9), 5) in (None, sum(range(10**5)))
