"""The point of an Order moved to infinity is solved by the native
solveset (it went to SymPy's solveset through the bridge); its first
solution is SymPy's, up to the form."""
from __future__ import annotations

import sympy

from sympy_extras._numeric import reliable_complex
from sympy_extras.core import Rational, Symbol
from sympy_extras.core.order import _solveset_first
from sympy_extras.interop import SymPyBridge, expr_from_sympy


def test_first_solution_matches_sympy() -> None:
    sk, sd = sympy.symbols('k d')
    k, d = Symbol('k'), Symbol('d')
    equations = [sk - sympy.exp(-sd), sk - 2*sympy.exp(3*sd), sympy.exp(sd) - 5, sympy.log(sd) - sk,
                 sympy.exp(1/sd) - sk]
    at = {k: Rational(3, 7)}
    native = 0
    for e in equations:
        ours = expr_from_sympy(e)
        solved, first = _solveset_first(ours, d)
        theirs = SymPyBridge().solve_first(ours, d)
        if not solved:
            # log(d) - k: the native solveset gives a ConditionSet, the
            # bridge's is asked
            continue
        native += 1
        if theirs is None:
            # SymPy's solveset gives a ConditionSet (exp(1/d) - k): the
            # native solution is checked by substitution
            assert first is not None
            residual = reliable_complex(ours.subs(d, first), 20, at)
            assert residual is not None and abs(residual) < 1e-15, (e, first)
            continue
        assert first is not None, e
        value = reliable_complex(first - theirs, 20, at)
        assert value is not None and abs(value) < 1e-15, (e, first, theirs)
    assert native == 4
    # an equation without solution, and one whose solutions are a union
    # (left to the bridge, which gives none)
    assert _solveset_first(expr_from_sympy(sympy.exp(sd)), d) == (True, None)
    assert _solveset_first(expr_from_sympy(sk - sympy.sin(sd)), d)[0] is False
