"""Tests of the native comprehensive Gröbner systems
(:mod:`sympy_extras.polys.comprehensive`) on the objects of the core: the
conditions found by :func:`exists_complex` are checked at rational
parameter values against the Gröbner bases of the specialised systems
(an independent computation: no branch, no condition), and the
eliminations and reductions against hand computations (no SymPy)."""
from __future__ import annotations

import random
from typing import Callable

from sympy_extras.core import Integer, Rational, Symbol, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.logic import Boolean, Not, and_, false, or_, to_boolean, true
from sympy_extras.core.relational import Eq, Ne, eq, ne
from sympy_extras.polys.comprehensive import (Branch, comprehensive_groebner_system, complex_quantifier_elimination,
    exists_complex, reduce_complex)
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.ideals import Ideal

a, b, c, x, y = [Symbol(name) for name in 'abcxy']


def raises(exception: type[BaseException], f: Callable[[], object]) -> None:
    try:
        f()
    except exception:
        return
    raise AssertionError('%s not raised' % exception.__name__)


def _solvable(polynomials: list[Expr], variables: list[Symbol]) -> bool:
    nonzero = [p for p in polynomials if p != Integer(0)]
    if any(not p.free_symbols for p in nonzero):
        return False
    return not nonzero or not Ideal(nonzero, *variables, domain=QQ).is_whole_ring()


def _check(equations: list[Expr], inequations: list[Expr], variables: list[Symbol], parameters: list[Symbol],
           rng: random.Random, samples: int = 10) -> None:
    formula = exists_complex(equations, inequations, variables)
    assert not formula.free_symbols & set(variables)
    values_list: list[dict[Basic, Basic]] = [
        {p: Rational(rng.randint(-3, 3), rng.choice([1, 2])) for p in parameters} for _ in range(samples)]
    values_list.append({p: Integer(0) for p in parameters})
    for values in values_list:
        specialised = [as_expr(e._xreplace_evaluated(values)) for e in equations]
        extra: list[Symbol] = []
        for i, g in enumerate(inequations):
            t = Symbol('z%d' % i)
            extra.append(t)
            specialised.append(t*as_expr(g._xreplace_evaluated(values)) - 1)
        expected = _solvable(specialised, variables + extra)
        actual = to_boolean(formula._xreplace_evaluated(values))
        assert actual in (true, false), (formula, values, actual)
        assert (actual == true) == expected, (equations, inequations, values, formula)


def test_comprehensive_groebner_system() -> None:
    branches = comprehensive_groebner_system([a*x - b], [x], [a, b])
    assert [repr(branch) for branch in branches] == ['Branch([], [a], [a*x - b])', 'Branch([a], [b], [1])',
                                                     'Branch([a, b], [], [])']
    assert [branch.solvable for branch in branches] == [True, False, True]
    assert branches[1].condition() == and_(eq(a, 0), ne(b, 0))
    assert Branch([], [], [Integer(1)]).solvable is False and Branch([], [], [x]).solvable is True


def test_exists_complex() -> None:
    assert exists_complex([a*x - b], [], [x]) == or_(ne(a, 0), and_(eq(a, 0), eq(b, 0)))
    assert exists_complex([x**2 + 1], [], [x]) == true
    assert exists_complex([x**2 + 1, x - 1], [], [x]) == false
    assert exists_complex([x**2 + 1], [x - a], [x]) == true
    assert exists_complex([x**2 - 2*x + 1], [x - a], [x]) == ne(a - 1, 0)
    assert exists_complex([], [x], [x]) == true and exists_complex([], [], [x]) == true
    assert exists_complex([x*y - 1], [], [x]) == ne(y, 0)
    rng = random.Random(5)
    systems: list[tuple[list[Expr], list[Expr], list[Symbol], list[Symbol]]] = [
        ([a*x - b], [], [x], [a, b]),
        ([a*x**2 + b*x + c, 2*a*x + b], [], [x], [a, b, c]),
        ([a*x**2 + b*x + c], [x], [x], [a, b, c]),
        ([x**2 - a, y**2 - b, x*y - c], [], [x, y], [a, b, c]),
        ([a*x + b*y - 1, x - y], [x], [x, y], [a, b]),
        ([x*y - a, x + y - b], [x - y], [x, y], [a, b]),
        ([a*x - 1, b*x - 1], [], [x], [a, b]),
    ]
    for equations, inequations, variables, parameters in systems:
        _check(equations, inequations, variables, parameters, rng)


def test_complex_quantifier_elimination() -> None:
    assert complex_quantifier_elimination(Eq(x**2, a), [('exists', x)]) == true
    assert complex_quantifier_elimination(Eq(a*x, 1), [('exists', x)]) == ne(a, 0)
    assert complex_quantifier_elimination(Eq(x*y, 1), [('forall', x), ('exists', y)]) == false
    assert complex_quantifier_elimination(Ne(x, y), [('forall', x), ('exists', y)]) == true
    # a double root of x**2 + a x + b
    assert complex_quantifier_elimination(Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0), [('exists', x)]) == \
        eq(a**2 - 4*b, 0)
    assert complex_quantifier_elimination(or_(Eq(x, a), Eq(x, b)) & Ne(x, a), [('exists', x)]) == ne(a - b, 0)
    assert complex_quantifier_elimination(and_(Not(Eq(x, a)), Not(Ne(x, a))), [('exists', x)]) == false
    # every x is a root of x**2 - a only when the polynomial is zero: never
    assert complex_quantifier_elimination(Eq(x**2, a), [('forall', x)]) == false


def test_reduce_complex() -> None:
    assert reduce_complex(Eq(x**2, 0) & Ne(x, 0)) == false
    assert reduce_complex(Eq(x, 1) & Eq(x, 2)) == false
    assert reduce_complex(Eq(x**2, 1) & Ne(x - 1, 0)) == and_(eq(x**2 - 1, 0), ne(x - 1, 0))
    assert reduce_complex(Ne(x**2 - 2*x + 1, 0)) == ne(x - 1, 0)
    # x*y = 1 and x = y: x**2 = 1, and x*y != 0 holds on the variety
    assert reduce_complex(Eq(x*y, 1) & Eq(x, y) & Ne(x*y, 0)) == and_(eq(x - y, 0), eq(y**2 - 1, 0))
    assert reduce_complex(true) == true
    raises(ValueError, lambda: reduce_complex(x > 0))
    raises(ValueError, lambda: reduce_complex(Eq(sqrt(x), 0)))
    raises(ValueError, lambda: reduce_complex(Eq(sqrt(Integer(2))*x, 1)))


def test_formulas_are_booleans_of_the_core() -> None:
    # the ideal <a*b> is radical and a does not vanish on it: nothing to
    # reduce (the conjunction is not decomposed into b = 0)
    result: Boolean = reduce_complex(Eq(a*b, 0) & Ne(a, 0))
    assert result == and_(eq(a*b, 0), ne(a, 0))
