"""The typed fibonacci, lucas, harmonic and bernoulli against SymPy (the
oracle), compared without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools

import pytest
import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import Integer, Rational, Symbol, oo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.functions import bernoulli, fibonacci, harmonic, lucas

x, n = Symbol('x'), Symbol('n')
k = Symbol('k', integer=True)


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def test_values_match_sympy() -> None:
    different = []
    integers: list[Expr] = [Integer(i) for i in range(-6, 13)]
    for a in integers + [oo, n, k, Rational(1, 2)]:
        sa = sympy_value(a)
        for ours, theirs in [(fibonacci(a), sympy.fibonacci(sa)), (lucas(a), sympy.lucas(sa)),
                             (harmonic(a), sympy.harmonic(sa)), (bernoulli(a), sympy.bernoulli(sa))]:
            if not same(ours, theirs):
                different.append((str(ours), str(theirs)))
    for a, b in itertools.product(integers[4:] + [oo, n], [Integer(2), Integer(0), Integer(-1), Integer(-2), Integer(1),
                                                            Rational(1, 2), Integer(3), k]):
        sa, sb = sympy_value(a), sympy_value(b)
        if not same(harmonic(a, b), sympy.harmonic(sa, sb)):
            different.append((str(harmonic(a, b)), str(sympy.harmonic(sa, sb))))
    for degree in range(0, 7):
        for point in [x, x + 1, Integer(1), Integer(0), Integer(-1), Rational(1, 2)]:
            ours, theirs = bernoulli(degree, point), sympy.bernoulli(degree, sympy_value(point))
            if not same(ours, theirs):
                different.append((str(ours), str(theirs)))
        if degree >= 1:
            if not same(fibonacci(degree, x), sympy.fibonacci(degree, sympy.Symbol('x'))):
                different.append((str(fibonacci(degree, x)), str(sympy.fibonacci(degree, sympy.Symbol('x')))))
    assert not different, different
    with pytest.raises(ValueError):
        fibonacci(0, x)


def test_derivatives_and_numbers() -> None:
    sn = sympy.Symbol('n')
    assert same(harmonic(n).diff(n), sympy.harmonic(sn).diff(sn))
    assert same(harmonic(n, 3).diff(n), sympy.harmonic(sn, 3).diff(sn))
    for e, s in [(harmonic(n).subs(n, Rational(1, 2)), sympy.harmonic(sympy.Rational(1, 2))),
                 (harmonic(n, Rational(1, 2)).subs(n, Rational(5, 2)), sympy.harmonic(sympy.Rational(5, 2), sympy.Rational(1, 2)))]:
        assert abs(complex(sympy_value(e.evalf(20))) - complex(sympy.N(s, 20))) < 1e-13, e


def test_euler_bell_genocchi_partition_match_sympy() -> None:
    from sympy_extras.core.numbers import Float
    from sympy_extras.functions import Genocchi, bell, euler, genocchi, partition
    different: list[tuple[str, ...]] = []
    m = Symbol('m', integer=True, nonnegative=True)
    indices: list[Expr] = [Integer(i) for i in range(-2, 15)] + [n, k, m, 2*m, 2*m + 1, Rational(1, 2), oo]
    for a in indices:
        sa = sympy_value(a)
        for ours_f, theirs_f in [(euler, sympy.euler), (bell, sympy.bell), (genocchi, sympy.genocchi),
                                 (partition, sympy.partition)]:
            try:
                theirs = theirs_f(sa)
            except (ValueError, TypeError) as error:
                with pytest.raises(type(error)):
                    ours_f(a)
                continue
            ours = ours_f(a)
            if not same(ours, theirs):
                different.append((ours_f.__name__, str(a), str(ours), str(theirs)))
    for degree in range(0, 8):
        for point in [x, x + 1, Integer(1), Integer(0), Rational(1, 2), Float('0.5')]:
            for ours_g, theirs_g in [(euler, sympy.euler), (bell, sympy.bell), (genocchi, sympy.genocchi)]:
                ours_value, theirs_value = ours_g(degree, point), theirs_g(degree, sympy_value(point))
                if not same(ours_value, theirs_value):
                    different.append((ours_g.__name__, str(degree), str(point), str(ours_value), str(theirs_value)))
    assert same(euler(Integer(-1), x), sympy.euler(-1, sympy.Symbol('x')))
    assert not different, different[:20]
    for a in [Integer(i) for i in range(0, 12)] + [m, 2*m, 2*m + 1, Integer(8)]:
        node = Genocchi(a)
        s = sympy.genocchi(sympy_value(a), evaluate=False)
        for name in ['integer', 'positive', 'negative', 'even', 'odd', 'prime']:
            got = node._ask(name)
            if got is not None:
                assert got == getattr(s, 'is_' + name), (node, name)
