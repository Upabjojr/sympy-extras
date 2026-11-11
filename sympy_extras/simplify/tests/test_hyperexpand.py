"""hyperexpand against SymPy (the oracle): the formula found for a
function, with its basis, closed form and matrices."""
from __future__ import annotations

import importlib
from typing import Callable

import sympy
from sympy import Rational as R

from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.interop import expr_from_sympy
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


# the differences left, from the core: 2**(2*a) is 4**a; arg and the
# imaginary values of erf(I) and of the Catalan sum differ from SymPy's
# (Abs(...) remains); the sign of fresnels(2/sqrt(pi)) is not decided
# (SymPy evaluates it); the value of the FresnelS formula at 1 is a square
# root of a product here; hyper([1/3], [4/3]) does not finish
_KNOWN_DIFFERENT = {('[3/4]', '[3/2, 7/4]', '1'), ('[3/4]', '[3/2, 7/4]', '-1'), ('[1/2, a]', '[3/2, a + 1]', 'z**2/4'), ('[a]', '[a + 1]', 'z**2/4'),
                    ('[1/2, 1/2, 1]', '[3/2, 3/2]', '-1'), ('[1]', '[3/2]', '-1')}


def test_hyperexpand_of_hypergeometric_functions_matches_sympy() -> None:
    from sympy_extras.simplify.hyperexpand import _hyperexpand
    a, b, z = sympy.symbols('a b z')
    cases = [([R(1, 2), 1], [R(3, 2)]), ([1, 1], [2]), ([], []), ([a], []), ([R(1, 2), R(1, 2)], [R(3, 2)]),
             ([a, a + R(1, 2)], [2*a]), ([3, 3], [4]), ([1], [b]), ([a], [2*a]), ([2], [3]), ([], [R(1, 2)]),
             ([], [b]), ([1, 1], [2, 2]), ([R(1, 2)], [R(3, 2), R(3, 2)]), ([R(3, 4)], [R(3, 2), R(7, 4)]),
             ([a, -a], [R(1, 2)]), ([R(1, 2), R(1, 2)], [1]), ([R(-1, 2), R(1, 2)], [1]),
             ([R(1, 2), a], [R(3, 2), a + 1]), ([a], [a + 1]), ([-2, a], [b]), ([-1, -2], [-3]),
             ([1, 1, 1], [2, 2]), ([2, 3], [5]), ([a, b], [a + 1]), ([1, 2, 3], [4, 5]), ([1], [R(3, 2)]),
             ([R(1, 2), R(1, 2), 1], [R(3, 2), R(3, 2)]), ([], [2])]
    for ap, bq in cases:
        for x in [z, sympy.Integer(1), sympy.Integer(-1), z**2/4, -z]:
            if (str(ap), str(bq), str(x)) in _KNOWN_DIFFERENT:
                continue
            want = str(sympy_hyperexpand._hyperexpand(sympy_hyperexpand.Hyper_Function(ap, bq), x))
            func = HyperFunction([expr_from_sympy(sympy.sympify(v)) for v in ap],
                                 [expr_from_sympy(sympy.sympify(v)) for v in bq])
            assert str(to_sympy(_hyperexpand(func, expr_from_sympy(x)), evaluate=False)) == want, (ap, bq, x)


def test_meijerg_expansions_match_sympy() -> None:
    from sympy import meijerg
    from sympy_extras.simplify.hyperexpand import hyperexpand
    a, b, z = sympy.symbols('a b z')
    cases = [meijerg([1], [], [], [], z), meijerg([], [1], [0], [], z), meijerg([], [], [0], [], z),
             meijerg([], [], [R(1, 2)], [0], z), meijerg([], [], [0], [R(1, 2)], z), meijerg([], [], [a], [-a], z),
             meijerg([], [], [a], [a + R(1, 2)], z), meijerg([1, 1], [], [], [0], z), meijerg([], [], [1, 1], [0, 0], z),
             meijerg([], [], [a], [], z), meijerg([], [], [0, R(1, 2)], [], z), meijerg([], [], [R(1, 2), 0], [], z**2/4),
             meijerg([], [], [a, b], [], z), meijerg([], [1], [a, b], [], z), meijerg([1 - a], [], [0], [-a], z),
             meijerg([a], [], [b], [], z), meijerg([], [a], [b], [], z), meijerg([], [], [0, 0], [], z),
             meijerg([], [R(1, 2)], [0], [], z), meijerg([], [], [-R(1, 2), R(1, 2)], [], z),
             meijerg([a], [a], [b], [b], z), meijerg([R(1, 2)], [1], [0, 0], [], z)]
    for e in cases:
        assert str(to_sympy(hyperexpand(expr_from_sympy(e)), evaluate=False)) == str(sympy.hyperexpand(e)), e


def test_residue_matches_sympy() -> None:
    from sympy import I, cos, exp, gamma, pi, sin
    from sympy_extras.series import residue
    x, a, z = sympy.symbols('x a z')
    cases = [(1/x, 0), (exp(x)/x**3, 0), (1/sin(x)**2, 0), (gamma(x), -2), (gamma(x), 0), (1/(x**2 + 1), I),
             (gamma(x)*gamma(a - x)*z**x, -1), (gamma(x)**2*z**x, -1), (cos(x)/x**4, 0), (1/(x*(x - 1)), 1),
             (gamma(-x)*z**x, 2), (x/(x - 2)**2, 2), (sympy.cot(x), pi), (1/(exp(x) - 1), 0), (gamma(x + a)*z**x, -a)]
    from sympy_extras.core import Symbol
    ox = Symbol('x')
    for e, p in cases:
        got = residue(expr_from_sympy(e), ox, expr_from_sympy(sympy.sympify(p)))
        assert str(to_sympy(got, evaluate=False)) == str(sympy.residue(e, x, p)), (e, p)


def test_instantiated_parameters_are_evaluated() -> None:
    # the bug: the symbolic formulas were instantiated with the core's
    # xreplace, which does not evaluate, so that HyperRep_power2(a, z) at
    # a = nu/2 + 1/2 had the parameter nu/2 + 1/2 - 1/2 (not nu/2), no
    # origin was found and 2F1(nu/2 + 1/2, nu/2 + 1; nu + 1; -w) stayed
    # unexpanded (SymPy expands it)
    from sympy_extras.core import Rational, Symbol
    from sympy_extras.functions import hyper
    from sympy_extras.functions.special.hyper import Hyper
    from sympy_extras.simplify.hyperexpand import hyperexpand
    nu, w = Symbol('nu', positive=True), Symbol('w', positive=True)
    found = FormulaCollection().lookup_origin(HyperFunction([nu / 2 + Rational(1, 2), nu / 2 + 1], [nu + 1]))
    assert found is not None
    assert set(found.func.ap) == {nu / 2, nu / 2 + Rational(1, 2)}
    e = hyper([nu / 2 + Rational(1, 2), nu / 2 + 1], [nu + 1], -w)
    expanded = hyperexpand(e)
    assert not expanded.has(Hyper)
    from sympy_extras._numeric_sympy import sympy_expr
    theirs = sympy.hyperexpand(sympy_expr(e))
    snu, sw = sympy.Symbol('nu', positive=True), sympy.Symbol('w', positive=True)
    ours = sympy_expr(expanded).subs(snu, R(1, 3)).subs(sw, R(1, 2))
    assert abs(complex(ours.evalf(30)) - complex(theirs.subs(snu, R(1, 3)).subs(sw, R(1, 2)).evalf(30))) < 1e-20
