"""hyperexpand against SymPy (the oracle): the formula found for a
function, with its basis, closed form and matrices."""
from __future__ import annotations

import importlib
from typing import Callable

import sympy
from sympy import Rational as R

from sympy_extras.core import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.core.interop import expr_from_sympy
from sympy_extras.matrices import ExprMatrix
from sympy_extras.simplify.hyperexpand import Formula, FormulaCollection, HyperFunction

sympy_hyperexpand = importlib.import_module('sympy.simplify.hyperexpand')


def _strs(m: ExprMatrix) -> list[str]:
    return [str(to_sympy(e, evaluate=False)) for e in m]


def _ours(f: Formula) -> tuple[str, list[str], list[str], list[str]]:
    params = str(tuple(to_sympy(e) for e in f.func.ap)) + str(tuple(to_sympy(e) for e in f.func.bq))
    return (params, _strs(f.B), _strs(f.C), _strs(f.M))


def test_lookup_origin_matches_sympy() -> None:
    a, b = sympy.symbols('a b')
    cases = [([R(1, 2), 1], [R(3, 2)]), ([1, 1], [2]), ([], []), ([a], []), ([R(1, 2), R(1, 2)], [R(3, 2)]),
             ([a, a + R(1, 2)], [2*a]), ([3, 3], [4]), ([R(1, 2), 2], [R(5, 2)]), ([1], [b]), ([1], [5]), ([a], [2*a]),
             ([2], [3]), ([], [R(1, 2)]), ([], [b]), ([1, 1], [2, 2]), ([R(1, 2)], [R(3, 2), R(3, 2)]),
             ([a, -a], [R(1, 2)]), ([R(1, 2), R(1, 2)], [1]), ([R(1, 2), a], [R(3, 2), a + 1]),
             ([1, 1, a], [2, 2, a + 1]), ([a], [a + 1]), ([R(1, 2)], [b, 2 - b]), ([1, 2], [3]), ([R(1, 3)], [R(4, 3)]),
             ([a, b], [a + b])]
    theirs_collection, ours_collection = sympy_hyperexpand.FormulaCollection(), FormulaCollection()
    for ap, bq in cases:
        theirs = theirs_collection.lookup_origin(sympy_hyperexpand.Hyper_Function(ap, bq))
        ap_ours: list[Expr] = [expr_from_sympy(sympy.sympify(x)) for x in ap]
        bq_ours: list[Expr] = [expr_from_sympy(sympy.sympify(x)) for x in bq]
        ours = ours_collection.lookup_origin(HyperFunction(ap_ours, bq_ours))
        if theirs is None:
            assert ours is None, (ap, bq)
            continue
        assert ours is not None, (ap, bq)
        want = (str(tuple(theirs.func.ap)) + str(tuple(theirs.func.bq)), [str(e) for e in theirs.B],
                [str(e) for e in theirs.C], [str(e) for e in theirs.M])
        assert _ours(ours) == want, (ap, bq)


def test_operators_and_reduce_order_match_sympy() -> None:
    import random
    from sympy_extras.simplify import hyperexpand as H
    a, b, z = sympy.symbols('a b z')

    def ours_list(values: list[sympy.Expr]) -> list[Expr]:
        return [expr_from_sympy(sympy.sympify(v)) for v in values]

    def attempt(make: Callable[[], list[str]]) -> list[str]:
        try:
            return make()
        except (ValueError, ZeroDivisionError):
            return ['error']

    def ours(op: H.Operator) -> list[str]:
        return [str(to_sympy(c, evaluate=False)) for c in op.coeffs]

    rng = random.Random(3)
    values = [R(1, 2), R(3, 2), 2, 3, R(-1, 2), a, a + 1, b, 2*a, R(1, 3), -a, a - R(1, 2)]
    zz = expr_from_sympy(z)
    for _ in range(30):
        ap = [rng.choice(values) for _ in range(rng.randint(1, 3))]
        bq = [rng.choice(values) for _ in range(rng.randint(1, 3))]
        i, j = rng.randrange(len(ap)), rng.randrange(len(bq))
        assert attempt(lambda: ours(H.UnShiftA(ours_list(ap), ours_list(bq), i, zz))) == attempt(
            lambda: [str(c) for c in sympy_hyperexpand.UnShiftA(ap, bq, i, z)._poly.all_coeffs()]), (ap, bq, i)
        assert attempt(lambda: ours(H.UnShiftB(ours_list(ap), ours_list(bq), j, zz))) == attempt(
            lambda: [str(c) for c in sympy_hyperexpand.UnShiftB(ap, bq, j, z)._poly.all_coeffs()]), (ap, bq, j)
        lists = [[rng.choice(values) for _ in range(rng.randint(1, 2))] for _ in range(4)]
        for name, k in [('MeijerUnShiftA', 2), ('MeijerUnShiftB', 0), ('MeijerUnShiftC', 3), ('MeijerUnShiftD', 1)]:
            idx = rng.randrange(len(lists[k]))
            ours_cls, theirs_cls = getattr(H, name), getattr(sympy_hyperexpand, name)
            assert attempt(lambda: ours(ours_cls(*[ours_list(v) for v in lists], idx, zz))) == attempt(
                lambda: [str(c) for c in theirs_cls(*lists, idx, z)._poly.all_coeffs()]), (name, lists, idx)
        theirs_func, theirs_ops = sympy_hyperexpand.reduce_order(
            sympy_hyperexpand.Hyper_Function(ap + [a + 2, 3], bq + [a, 1]))
        ours_func, ours_ops = H.reduce_order(H.HyperFunction(ours_list(ap + [a + 2, 3]), ours_list(bq + [a, 1])))
        assert str(tuple(to_sympy(x) for x in ours_func.ap)) == str(tuple(theirs_func.ap))
        assert str(tuple(to_sympy(x) for x in ours_func.bq)) == str(tuple(theirs_func.bq))
        assert [[str(to_sympy(c, evaluate=False)) for c in op.coeffs] for op in ours_ops] == \
            [[str(c) for c in op._poly.all_coeffs()] for op in theirs_ops]
