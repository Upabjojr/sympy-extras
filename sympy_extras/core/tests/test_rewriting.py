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
