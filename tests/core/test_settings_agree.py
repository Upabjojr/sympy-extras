"""``sympy_extras.settings`` against its Rust port: the defaults and the
settings inside and after ``configure``. The Rust ``configure`` checks
its arguments before it changes anything; the Python leaves the settings
given before an invalid ``groebner_algorithm`` changed (restored here by
hand)."""

from __future__ import annotations

from typing import Any, Optional

from ..conftest import load


def _python(s: Any) -> tuple[Any, ...]:
    return (s.numerical_checks, s.precision, s.timeout, s.time_scale, s.modular_groebner, s.groebner_direct_time,
            s.groebner_algorithm)


def _rust(t: Any) -> tuple[Any, ...]:
    a = t.args
    timeout: Optional[float] = None if not a[2].args and str(a[2]) == '()' else float(str(a[2]))
    return (bool(a[0]), int(a[1]), timeout, float(str(a[3])), bool(a[4]), float(str(a[5])), str(a[6]))


def _arg(v: Any) -> Any:
    from sympy_extras.core import Integer, Symbol, Tuple
    from sympy_extras.core.logic import false, true
    if v is None:
        return Tuple()
    if isinstance(v, bool):
        return true if v else false
    if isinstance(v, int):
        return Integer(v)
    return Symbol(str(v))


def test_defaults(rust: Any) -> None:
    from sympy_extras.settings import settings
    assert _rust(load("sympy_extras_rs.core").logic_call('settings', [])) == _python(settings)


def test_configure(rust: Any) -> None:
    from sympy_extras.settings import Settings, configure, settings
    rc = load("sympy_extras_rs.core")
    cases = [dict(numerical_checks=False), dict(precision=50, timeout=5.0), dict(no_timeout=True),
             dict(groebner_algorithm='buchberger', time_scale=2.5), dict(numerical_checks=False, groebner_algorithm='x'),
             dict(time_scale=0.0)]
    keys = ['numerical_checks', 'precision', 'timeout', 'no_timeout', 'groebner_algorithm', 'time_scale']
    for case in cases:
        before = _python(settings)
        args = [_arg(case.get(k, False if k == 'no_timeout' else None)) for k in keys]
        inside, after = rc.logic_call('settings', args).args
        try:
            with configure(**case) as s:
                expected = _python(s)
        except ValueError:
            expected = None
            # the Python leaves its changes before the error: restored by hand
            fresh = Settings()
            for attribute in ('numerical_checks', 'precision', 'timeout', 'modular_groebner', 'groebner_direct_time',
                              'groebner_algorithm'):
                setattr(settings, attribute, getattr(fresh, attribute))
        if expected is None:
            assert str(inside).startswith('(ERROR'), case
        else:
            assert _rust(inside) == expected, case
        assert _rust(after) == before == _python(settings), case
