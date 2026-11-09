"""The closed forms of the HyperRep functions (hyperexpand's tables)
against SymPy's rewrites 'nonrepsmall' and 'nonrep', on several branches
of the argument."""
from __future__ import annotations

from typing import Optional

import sympy
import sympy.functions.special.hyper as sympy_hyper

from sympy_extras.core import I, Rational, Symbol, pi
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.interop import expr_from_sympy
from sympy_extras.functions.elementary.exponential import exp_polar
from sympy_extras.functions.special import hyperrep


def test_closed_forms_match_sympy() -> None:
    a, z = Symbol('a'), Symbol('z')
    branches = [Rational(0), Rational(1), Rational(-1), Rational(2), Rational(1, 2), Rational(-1, 2), Rational(3, 2)]
    different: list[str] = []
    for name in hyperrep.__all__:
        if not name.startswith('HyperRep_'):
            continue
        native, theirs = getattr(hyperrep, name), getattr(sympy_hyper, name)
        parameters: list[Optional[Expr]] = [a, Rational(1, 3), Rational(2)] if 2 in native.nargs else [None]
        for k in branches:
            arg = z if k == 0 else z * exp_polar(2*pi*I*k)
            for p in parameters:
                ours = native(arg) if p is None else native(p, arg)
                sympy_node = theirs(*[to_sympy(x) for x in ours.args])
                assert type(expr_from_sympy(sympy_node)) is native
                for small in (True, False):
                    want = str(sympy_node.rewrite('nonrepsmall' if small else 'nonrep'))
                    got = str(to_sympy(hyperrep.rewrite_nonrep(ours, small), evaluate=False))
                    if want != got:
                        different.append('%s %s %s: %s vs %s' % (ours, k, small, got, want))
    assert not different, different
