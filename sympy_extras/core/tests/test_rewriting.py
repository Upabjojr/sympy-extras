"""Oracle tests of the native rewrite (exp and cos targets) against SymPy."""
import random

import sympy

from sympy_extras.core.interop import expr_from_sympy, to_sympy
from sympy_extras.core.mul import neg
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.functions.elementary.trigonometric import Cos

s = sympy
x, y = s.symbols('x y')
p = s.Symbol('p', positive=True)


def test_rewrite_exp_and_cos_match_sympy() -> None:
    rng = random.Random(2)
    fs = [s.sin, s.cos, s.tan, s.cot, s.sec, s.csc, s.sinh, s.cosh, s.tanh, s.coth, s.sech, s.csch, s.exp, s.log,
          s.sqrt, s.gamma, s.asin, lambda a: a**2, lambda a: 2**a, lambda a: 1 / a]
    for _ in range(80):
        e = rng.choice([x, y, p, 2 * x, x + y, s.pi / 3, s.sqrt(3) / 2])
        for _ in range(rng.randint(1, 3)):
            e = rng.choice(fs)(e)
        for theirs, ours in ((s.exp, Exp), (s.cos, Cos)):
            assert str(to_sympy(expr_from_sympy(e).rewrite(ours), evaluate=False)) == str(e.rewrite(theirs)), e


def test_neg_does_not_flatten_like_sympy() -> None:
    # SymPy's __neg__ builds Mul._from_args: an unevaluated exp stays
    u = s.exp(2 * s.log(p), evaluate=False)
    assert str(to_sympy(neg(expr_from_sympy(u)), evaluate=False)) == str(-u)
    for e in (x, x + y, -2 * x, 2 * x * y, s.I * x, s.zoo * x):
        assert str(to_sympy(neg(expr_from_sympy(e)), evaluate=False)) == str(-e)


def test_count_ops_matches_sympy() -> None:
    from sympy_extras.core.function import count_ops
    rng = random.Random(9)
    atoms = [x, y, s.Integer(2), s.Rational(-1, 2), s.pi, -x]
    fs = [s.sin, s.exp, s.log, s.sqrt, lambda a: a**2, lambda a: 1 / a, lambda a: -a, lambda a: a**s.Rational(-3, 2)]

    def rnd(d: int) -> sympy.Expr:
        if d == 0 or rng.random() < .2:
            return rng.choice(atoms)
        q = rng.random()
        if q < .4:
            return rnd(d - 1) + rnd(d - 1)
        if q < .75:
            return rnd(d - 1) * rnd(d - 1)
        return rng.choice(fs)(rnd(d - 1))
    for _ in range(300):
        e = rnd(5)
        for visual in (False, True):
            assert str(count_ops(expr_from_sympy(e), visual=visual)) == str(s.count_ops(e, visual=visual)), e


def test_traversals_rebuild_only_changed_nodes() -> None:
    from sympy_extras.core.function import expand_log
    from sympy_extras.simplify.simplify import logcombine
    e = s.factor_terms(2 * y * s.log(2) + 4, sign=False)  # 2*(y*log(2) + 2)
    assert str(to_sympy(expand_log(expr_from_sympy(e), deep=True), evaluate=False)) == str(s.expand_log(e, deep=True))
    assert str(to_sympy(logcombine(expr_from_sympy(e)), evaluate=False)) == str(s.logcombine(e))


def test_add_recreates_nested_terms_and_normal_keeps_numeric_denominator() -> None:
    from sympy_extras.core.add import add
    nested = s.Mul._from_args((s.S.Half, x * y * (2 * y + 1)))
    ours = add(expr_from_sympy(nested), expr_from_sympy(y**2))
    assert [a.args for a in ours.args] == [tuple(expr_from_sympy(b) for b in a.args) for a in (nested + y**2).args]
    e = x * y * (y + s.S.Half) + y**2
    assert str(to_sympy(expr_from_sympy(e).normal(), evaluate=False)) == str(e.normal())


def test_rewrite_gamma_matches_sympy() -> None:
    from sympy_extras.functions.special.gamma_functions import Gamma
    n, k = s.symbols('n k', integer=True)
    m = s.Symbol('m', negative=True)
    cases: list[sympy.Expr] = [
        s.factorial(n) / s.factorial(n - 3), s.binomial(n, k), s.rf(x, k), s.ff(x, 3), s.rf(m, y)**2, s.ff(p, k),
        s.factorial2(n)**2, s.subfactorial(x), s.catalan(n), s.beta(x, p), s.rf(k, k)**2 + s.factorial(k)]
    for e in cases:
        for piecewise in (True, False):
            ours = to_sympy(expr_from_sympy(e).rewrite(Gamma, piecewise=piecewise), evaluate=False)
            assert str(ours) == str(e.rewrite(s.gamma, piecewise=piecewise)), (e, piecewise)


def test_expand_complex_matches_sympy() -> None:
    from sympy_extras.core.function import expand, expand_complex
    r = s.Symbol('r', real=True)
    cases: list[sympy.Expr] = [
        x + s.I * y, s.exp(x), s.sin(2 * p * x**2), s.log(p + y), y / (y + s.I), s.sqrt(x) + 1, p * (s.sqrt(x) + 1),
        s.cosh(r) * s.sqrt(s.Abs(x)), r**2 + x**2 + s.log(p + y), s.log(s.pi * r * x * (1 + s.I)), s.tan(s.pi * x * y / 2),
        2 * s.Abs(y)]
    for e in cases:
        assert str(to_sympy(expand_complex(expr_from_sympy(e)), evaluate=False)) == str(s.expand_complex(e)), e
        assert str(to_sympy(expand(expr_from_sympy(e), complex=True), evaluate=False)) == str(s.expand(e, complex=True)), e
