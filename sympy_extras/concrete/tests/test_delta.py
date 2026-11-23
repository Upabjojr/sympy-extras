"""Sums and products with Kronecker deltas against SymPy (the oracle)."""
from __future__ import annotations

import sympy
from sympy import KroneckerDelta as KD, oo
from sympy.concrete.delta import deltaproduct as sympy_deltaproduct, deltasummation as sympy_deltasummation

from sympy_extras.concrete.delta import deltaproduct, deltasummation
from sympy_extras.concrete.products import product
from sympy_extras.concrete.summations import summation
from sympy_extras.interop import to_sympy
from sympy_extras.interop import expr_from_sympy, from_sympy
from sympy_extras.core.symbol import Symbol

i, j, k, n = sympy.symbols('i j k n', integer=True)
x = sympy.Symbol('x')
FS = [KD(i, k), k*KD(i, k), KD(i, 2*k), KD(i, k)*KD(j, k), x**k*KD(k, 3), KD(i, k) + k, (KD(i, k) + 1)*k,
      KD(i, k + 1)*k**2, KD(k, 2) + KD(k, 3), KD(i, j)*k]
LIMITS = [(k, 0, n), (k, 1, 3), (k, 1, n), (k, -oo, oo), (k, 0, 5)]


def test_delta_sums_and_products_match_sympy() -> None:
    for f in FS:
        for v, lo, hi in LIMITS:
            slo, shi = sympy.sympify(lo), sympy.sympify(hi)
            var = from_sympy(v)
            assert isinstance(var, Symbol)
            limits = (var, expr_from_sympy(slo), expr_from_sympy(shi))
            of = expr_from_sympy(f)
            if f != KD(i, 2*k):
                # (SymPy's sum of KroneckerDelta(i, 2*k) over k is 1 for an odd
                # i in the range too, sympy/sympy#31275, V47: see
                # test_delta_sum_asks_for_a_solution_on_the_lattice_of_the_index)
                assert str(to_sympy(deltasummation(of, limits), evaluate=False)) == str(
                    sympy_deltasummation(f, (v, slo, shi)))
                assert str(to_sympy(summation(of, limits), evaluate=False)) == str(sympy.summation(f, (v, slo, shi)))
            if lo == -oo:
                continue
            assert str(to_sympy(deltaproduct(of, limits), evaluate=False)) == str(sympy_deltaproduct(f, (v, slo, shi)))
            assert str(to_sympy(product(of, limits), evaluate=False)) == str(sympy.product(f, (v, slo, shi)))


def test_canonical_orders_by_nodes_as_sympy() -> None:
    from sympy_extras.core.relational import Relational
    for r in [i/2 <= n, n >= i/2, x + 1 < 2*n, -x <= n, i < x**2 + 1]:
        ours = from_sympy(r)
        assert isinstance(ours, Relational)
        assert str(to_sympy(ours.canonical, evaluate=False)) == str(r.canonical), r


def test_delta_solutions_respect_the_assumptions_of_the_index() -> None:
    # the native solve of pi*k = 1 gives {1/pi} for an integer k; taken
    # as the point of the delta, the sum was Piecewise((1/pi, n >= 1/pi),
    # (0, True)) where SymPy's solve (dropping 1/pi) gives 0
    from sympy_extras.core import Integer, pi
    from sympy_extras.functions.special.tensor_functions import KroneckerDelta
    k, n = Symbol('k', integer=True), Symbol('n', integer=True)
    assert deltasummation(k*KroneckerDelta(pi*k, 1), (k, Integer(0), n)) == 0


def test_delta_sum_asks_for_a_solution_on_the_lattice_of_the_index() -> None:
    # sympy/sympy#31275, V47: the index of a sum takes integer steps, and
    # KroneckerDelta(x, y) is 0 on all of them for y = 1/2; SymPy only asks
    # for y in the range: its Sum(KroneckerDelta(x, 1/2), (x, -oo, oo)) is 1
    from sympy_extras.core import Integer, Rational, oo as inf
    from sympy_extras.core.logic import and_, true
    from sympy_extras.functions import KroneckerDelta
    from sympy_extras.functions.elementary.piecewise import piecewise
    from sympy_extras.sets.fancysets import Integers
    from sympy_extras.sets.sets import contains
    u, v, m = Symbol('u'), Symbol('v'), Symbol('m')
    q = Symbol('q', integer=True)
    assert summation(KroneckerDelta(u, Rational(1, 2)), (u, -inf, inf)) == 0
    assert summation(KroneckerDelta(u, Rational(1, 2)), (u, Integer(-3), Integer(3))) == 0
    assert summation(KroneckerDelta(u, Integer(2)), (u, -inf, inf)) == 1
    general = summation(m*KroneckerDelta(u, v), (u, -inf, inf))
    assert general == piecewise((m, and_(v > -inf, v < inf, contains(v, Integers))), (Integer(0), true))
    assert general.subs(v, Rational(1, 2)) == 0 and general.subs(v, 7) == m
    # an integer solution: the range alone, as before
    assert summation(KroneckerDelta(u, q), (u, -inf, inf)) == piecewise(
        (Integer(1), and_(q > -inf, q < inf)), (Integer(0), true))
    # a symbolic lower limit: the lattice is a + integers
    a = Symbol('a')
    shifted = deltasummation(KroneckerDelta(u, v), (u, a, inf))
    assert shifted.subs({v: Rational(5, 2), a: Rational(1, 2)}) == 1
    assert shifted.subs({v: Rational(5, 2), a: Integer(1)}) == 0
    # an odd integer is no 2*k
    p, w = Symbol('p', integer=True), Symbol('w', integer=True)
    halves = deltasummation(KroneckerDelta(p, 2*w), (w, Integer(0), Integer(5)))
    assert halves.subs(p, 3) == 0 and halves.subs(p, 4) == 1 and halves.subs(p, 12) == 0
