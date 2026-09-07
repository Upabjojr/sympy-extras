from __future__ import annotations

import random

from sympy import S, Eq, Ne, And, Or, Not, Rational, Symbol
from sympy.abc import a, b, c, x, y
from sympy.core.basic import Basic
from sympy.core.expr import Expr

from sympy_extras._typing import as_expr
from sympy.testing.pytest import raises

from sympy_extras.assumptions import resolve, ForAll, Exists
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.comprehensive import (comprehensive_groebner_system, exists_complex,
    complex_quantifier_elimination, reduce_complex, Branch)


def _solvable(polynomials: list[Expr], variables: list[Symbol]) -> bool:
    nonzero = [p for p in polynomials if p != 0]
    if any(p.is_number for p in nonzero):
        return False
    return not nonzero or not Ideal(nonzero, *variables).is_whole_ring()


def _check(equations: list[Expr], inequations: list[Expr], variables: list[Symbol],
           parameters: list[Symbol], rng: random.Random, samples: int = 12) -> None:
    formula = exists_complex(equations, inequations, variables)
    assert not formula.has(*variables)
    values_list: list[dict[Basic, Basic]] = [
        {p: Rational(rng.randint(-3, 3), rng.choice([1, 2])) for p in parameters} for _ in range(samples)]
    # also the special values where the coefficients may vanish
    values_list += [{p: S.Zero for p in parameters}]
    for values in values_list:
        specialised = [as_expr(e.xreplace(values)) for e in equations]
        z_ = []
        for i, g in enumerate(inequations):
            t = Symbol('z%d' % i)
            z_.append(t)
            specialised.append(as_expr(t*g.xreplace(values) - 1))
        expected = _solvable(specialised, variables + z_)
        actual = formula.xreplace(values)
        assert actual in (S.true, S.false), (formula, values, actual)
        assert (actual is S.true) == expected, (equations, inequations, values, formula)


def test_comprehensive_groebner_system() -> None:
    branches = comprehensive_groebner_system([a*x - b], [x], [a, b])
    assert [(br.equations, br.nonzero, br.basis) for br in branches] == \
        [([], [a], [a*x - b]), ([a], [b], [1]), ([a, b], [], [])]
    assert isinstance(branches[0], Branch)
    assert branches[0].solvable and not branches[1].solvable and branches[2].solvable
    assert branches[0].condition() == Ne(a, 0)
    # the classical example: on each branch the basis specialises to a Groebner basis
    branches = comprehensive_groebner_system([a*x**2 + b*x + c], [x], [a, b, c])
    assert any(br.equations == [a] and b in br.nonzero for br in branches)
    assert any(br.equations == [a, b] and c in br.nonzero and br.basis == [1] for br in branches)
    assert any(br.equations == [a, b, c] and br.basis == [] for br in branches)


def test_exists_complex() -> None:
    assert exists_complex([a*x - b], [], [x]) == Or(Ne(a, 0), And(Eq(a, 0), Eq(b, 0)))
    assert exists_complex([x**2 + 1], [], [x]) is S.true
    assert exists_complex([x**2 + 1, x - 1], [], [x]) is S.false
    assert exists_complex([x**2 + 1], [x - a], [x]) is S.true
    assert exists_complex([x**2 - 2*x + 1], [x - a], [x]) == Ne(a - 1, 0)
    assert exists_complex([], [x], [x]) is S.true
    assert exists_complex([], [], [x]) is S.true
    assert exists_complex([x*y - 1], [], [x]) == Ne(y, 0)
    rng = random.Random(3)
    systems = [
        ([a*x - b], [], [x], [a, b]),
        ([a*x**2 + b*x + c, 2*a*x + b], [], [x], [a, b, c]),
        ([a*x**2 + b*x + c], [x], [x], [a, b, c]),
        ([x**2 - a, y**2 - b, x*y - c], [], [x, y], [a, b, c]),
        ([x**2 - a, x - b], [], [x], [a, b]),
        ([a*x + b*y - 1, x - y], [x], [x, y], [a, b]),
        ([x**3 - a*x - b, 3*x**2 - a], [], [x], [a, b]),
        ([x*y - a, x + y - b], [x - y], [x, y], [a, b]),
        ([a*x**2 + b*x + c], [x - 1], [x], [a, b, c]),
    ]
    for equations, inequations, variables, parameters in systems:
        _check(equations, inequations, variables, parameters, rng)


def test_exists_complex_random() -> None:
    rng = random.Random(11)
    for _ in range(15):
        equations: list[Expr] = []
        for _ in range(rng.randint(1, 2)):
            e = sum((rng.choice([a, b, 1, 2, a - 1]) * x**i for i in range(rng.randint(1, 3))), S.Zero)
            equations.append(as_expr(e))
        inequations = [x - rng.choice([a, b, 1])] if rng.random() < 0.4 else []
        _check(equations, inequations, [x], [a, b], rng, samples=8)


def test_complex_quantifier_elimination() -> None:
    assert complex_quantifier_elimination(Eq(x**2, a), [('exists', x)]) is S.true
    assert complex_quantifier_elimination(Eq(a*x, 1), [('exists', x)]) == Ne(a, 0)
    assert complex_quantifier_elimination(Eq(x*y, 1), [('forall', x), ('exists', y)]) is S.false
    assert complex_quantifier_elimination(Eq(x*y, 1), [('exists', y), ('forall', x)]) is S.false
    assert complex_quantifier_elimination(Ne(x, y), [('forall', x), ('exists', y)]) is S.true
    assert complex_quantifier_elimination(Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0), [('exists', x)]) == Eq(a**2 - 4*b, 0)
    assert complex_quantifier_elimination(Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0), [('forall', x)]) is S.false
    # every polynomial of positive degree has a root: the fundamental theorem
    assert complex_quantifier_elimination(Eq(x**3 + a*x + b, 0), [('exists', x)]) is S.true
    assert complex_quantifier_elimination(Or(Eq(x, a), Eq(x, b)) & Ne(x, a), [('exists', x)]) == Ne(a - b, 0)
    assert complex_quantifier_elimination(Not(Eq(x, a)) & Not(Ne(x, a)), [('exists', x)]) is S.false


def test_reduce_complex() -> None:
    assert reduce_complex(Eq(x**2, 0) & Ne(x, 0)) is S.false
    assert reduce_complex(Eq(x**2 - y, 0) & Eq(x*y - 1, 0) & Ne(x, 0)) == \
        And(Eq(x*y - 1, 0), Eq(-x + y**2, 0), Eq(x**2 - y, 0))
    assert reduce_complex(Eq(x, 1) & Eq(x, 2)) is S.false
    assert reduce_complex(Eq(x**2, 1) & Ne(x - 1, 0)) == And(Eq(x**2 - 1, 0), Ne(x - 1, 0))
    assert reduce_complex(Ne(x, 0) | Eq(x, 0)) == Or(Ne(x, 0), Eq(x, 0))
    assert reduce_complex(Ne(x**2 - 2*x + 1, 0)) == Ne(x - 1, 0)
    assert reduce_complex(S.true) is S.true
    raises(ValueError, lambda: reduce_complex(x > 0))
    raises(ValueError, lambda: reduce_complex(Eq(x**Rational(1, 2), 0)))


def test_resolve_complexes() -> None:
    assert resolve(Exists(x, Eq(x**2 + 1, 0)), domain=S.Complexes) is S.true
    assert resolve(Exists(x, Eq(x**2 + 1, 0)), domain=S.Reals) is S.false
    assert resolve(Exists(x, Eq(a*x, 1)), domain=S.Complexes) == Ne(a, 0)
    assert resolve(ForAll(x, Exists(y, Eq(x*y, 1))), domain=S.Complexes) is S.false
    assert resolve(ForAll(x, Exists(y, Eq(y**2, x))), domain=S.Complexes) is S.true
    assert resolve(Exists(x, Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0)), domain=S.Complexes) == Eq(a**2 - 4*b, 0)
    assert resolve(Eq(x**2, 0) & Ne(x, 0), domain=S.Complexes) is S.false
    assert resolve(Exists(x, Eq(a*x, 1)), domain=S.Complexes, assumptions=Ne(a, 0)) is S.true
    raises(ValueError, lambda: resolve(Exists(x, x > 0), domain=S.Complexes))
