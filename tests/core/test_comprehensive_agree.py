"""``sympy_extras.polys.comprehensive`` against its Rust port:
comprehensive Gröbner systems (Kapur–Sun–Wang), the existential closure
of systems of equations and inequations over the complex numbers,
quantifier elimination for Boolean combinations of polynomial equations
and inequations (all the connectives through the disjunctive normal
form), and the simplification by Gröbner bases."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _symbols() -> Any:
    from sympy_extras.core import symbols
    return symbols('x y a b c')


def _cgs_cases() -> list[tuple[list[Any], list[Any], list[Any]]]:
    x, y, a, b, c = _symbols()
    return [([a*x - b], [x], [a, b]), ([a*x**2 + b*x + c], [x], [a, b, c]), ([a*x + y, b*y - 1], [x, y], [a, b]),
            ([x**2 - a, x*y - b], [x, y], [a, b]), ([a*x**2 - 1, x - b], [x], [a, b]), ([x*y - a, x + y - b], [x, y], [a, b])]


@pytest.mark.parametrize("index", range(6))
def test_cgs(rust: Any, index: int) -> None:
    from sympy_extras.core import Symbol, Tuple
    from sympy_extras.polys.comprehensive import comprehensive_groebner_system
    rc = load("sympy_extras_rs.core")
    polys, variables, parameters = _cgs_cases()[index]

    def got() -> Any:
        r = rc.logic_call('complex_reduction', [Symbol('cgs'), Tuple(*polys), Tuple(*variables), Tuple(*parameters)])
        return [[[str(e) for e in part.args] for part in b.args] for b in r.args]

    def expected() -> Any:
        return [[[str(e) for e in part] for part in (b.equations, b.nonzero, b.basis)]
                for b in comprehensive_groebner_system(polys, variables, parameters)]
    assert outcome(got) == outcome(expected), polys


def _exists_cases() -> list[tuple[list[Any], list[Any], list[Any]]]:
    x, y, a, b, c = _symbols()
    return [([a*x - b], [], [x]), ([x**2 + 1], [], [x]), ([x**2 + 1, x - 1], [], [x]), ([x**2 + 1], [x - a], [x]),
            ([x**2 - 2*x + 1], [x - a], [x]), ([], [x], [x]), ([], [], [x]), ([x*y - 1], [], [x]),
            ([a*x**2 + b*x + c, 2*a*x + b], [a], [x]), ([x*y - a, x - y], [x - b], [x, y])]


@pytest.mark.parametrize("index", range(10))
def test_exists(rust: Any, index: int) -> None:
    from sympy_extras.core import Symbol, Tuple
    from sympy_extras.polys.comprehensive import exists_complex
    rc = load("sympy_extras_rs.core")
    equations, inequations, variables = _exists_cases()[index]
    assert outcome(lambda: str(rc.logic_call('complex_reduction', [Symbol('exists'), Tuple(*equations),
                                                                    Tuple(*inequations), Tuple(*variables)]))) == \
        outcome(lambda: str(exists_complex(equations, inequations, variables))), (equations, inequations)


def _qe_cases() -> list[tuple[Any, list[tuple[str, Any]]]]:
    from sympy_extras.core import Eq, Equivalent, ITE, Implies, Ne, Not, Or, Xor
    x, y, a, b, c = _symbols()
    return [(Eq(x**2, a), [('exists', x)]), (Eq(a*x, 1), [('exists', x)]), (Eq(x*y, 1), [('forall', x), ('exists', y)]),
            (Eq(x*y, 1), [('exists', y), ('forall', x)]), (Ne(x, y), [('forall', x), ('exists', y)]),
            (Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0), [('exists', x)]),
            (Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0), [('forall', x)]), (Eq(x**3 + a*x + b, 0), [('exists', x)]),
            (Or(Eq(x, a), Eq(x, b)) & Ne(x, a), [('exists', x)]), (Not(Eq(x, a)) & Not(Ne(x, a)), [('exists', x)]),
            (Implies(Eq(a*x, 1), Eq(x, b)), [('forall', x)]), (Equivalent(Eq(x, a), Eq(x, b)), [('forall', x)]),
            (Xor(Eq(x, a), Eq(x, b)), [('exists', x)]), (ITE(Eq(a, 0), Eq(x, 1), Eq(a*x, b)), [('exists', x)])]


@pytest.mark.parametrize("index", range(14))
def test_quantifier_elimination(rust: Any, index: int) -> None:
    from sympy_extras.core import Symbol, Tuple
    from sympy_extras.polys.comprehensive import complex_quantifier_elimination
    rc = load("sympy_extras_rs.core")
    formula, prefix = _qe_cases()[index]
    rust_prefix = Tuple(*[Tuple(Symbol(k), v) for k, v in prefix])
    assert outcome(lambda: str(rc.logic_call('complex_reduction', [Symbol('qe'), formula, rust_prefix]))) == \
        outcome(lambda: str(complex_quantifier_elimination(formula, prefix))), formula


def test_reduce(rust: Any) -> None:
    from sympy_extras.core import Eq, Lt, Ne, Symbol
    from sympy_extras.polys.comprehensive import reduce_complex
    rc = load("sympy_extras_rs.core")
    x, y, a, b, c = _symbols()
    for formula in [Eq(x**2, 0) & Ne(x, 0), Eq(x**2 - y, 0) & Eq(x*y - 1, 0) & Ne(x, 0), Eq(x, 1) & Eq(x, 2),
                    Eq(x**2, 1) & Ne(x - 1, 0), Ne(x, 0) | Eq(x, 0), Eq(x*y, 0) & Ne(x**2*y + x, 0),
                    (Eq(x, 1) | Eq(y, 2)) & Ne(x*y - 2, 0), Lt(x, 1), Eq(x, Symbol('w')**Symbol('w'))]:
        assert outcome(lambda: str(rc.logic_call('complex_reduction', [Symbol('reduce'), formula]))) == \
            outcome(lambda: str(reduce_complex(formula))), formula
