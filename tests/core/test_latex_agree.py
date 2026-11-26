"""The LaTeX printer of the Rust core (``printing/latex.rs``) against
``sympy_extras.printing.latex``: numbers, names, sums, products, powers,
functions, relations, logic, derivatives, with the settings of the
printer; the nodes not ported yet are a skip on the Rust side."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed

SETTINGS: list[dict[str, Any]] = [
    {}, {'mode': 'inline'}, {'mode': 'equation*'}, {'fold_frac_powers': True}, {'fold_func_brackets': True},
    {'fold_short_frac': True}, {'mul_symbol': 'dot'}, {'mul_symbol': 'times'}, {'ln_notation': True},
    {'inv_trig_style': 'full'}, {'inv_trig_style': 'power'}, {'root_notation': False}, {'itex': True},
    {'imaginary_unit': 'ri'}, {'gothic_re_im': True}, {'decimal_separator': 'comma'},
    {'parenthesize_super': False}, {'diff_operator': 'rd'}, {'long_frac_ratio': 2.0}, {'order': 'none'},
]


def _outcome(f: Any) -> Any:
    try:
        return ('ok', f())
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _cases() -> list[Any]:
    from sympy_extras.core import (E, Eq, Function, GoldenRatio, I, Integer, Ne, Rational, Symbol, oo, pi, sqrt, zoo,
                                   nan)
    from sympy_extras.core.function import Derivative, Subs
    from sympy_extras.core.mod import mod
    from sympy_extras.core.logic import And, Equivalent, Implies, Not, Or, Xor
    from sympy_extras import functions as f
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    n = Symbol('n', integer=True)
    alpha, xdot, x1 = Symbol('alpha'), Symbol('xdot'), Symbol('x_1^2')
    F, G = Function('f'), Function('Gamma_1')
    return [Integer(3), Integer(-4), Rational(1, 2), Rational(-3, 4), oo, -oo, zoo, nan, pi, E, I, GoldenRatio,
            alpha, xdot, x1, Symbol('Omega'), Symbol('lamda'), Symbol('a%b'), Symbol('xhatbold'),
            x + y, x - y, 2*x, -x, x*y/z, (x + 1)**2/y, x**Rational(2, 3), sqrt(x), x**Rational(-1, 3), 1/x,
            1/(x + 1), x**-2, 2**x, (-2)**x, x**(y + 1), x**y**z, -x**2, 3*x/2, -Rational(1, 2)*x, x*2**y,
            f.sin(x)**2, f.sin(x + y), f.cos(2*x), f.exp(x), f.exp(-x)*y, f.log(x), f.log(x)**2, f.Abs(x - 1),
            f.floor(x), f.ceiling(x/2), f.asin(x), f.acosh(x)**2, f.atan(x), f.gamma(x)**2, f.besselj(n, x),
            f.hankel1(n, x)**2, f.factorial(n), f.factorial(x + 1), f.binomial(n, 2), f.RisingFactorial(x, n),
            f.FallingFactorial(x, n), f.zeta(x), f.polylog(2, x), f.expint(n, x), f.legendre(n, x),
            f.jacobi(n, x, y, z), f.assoc_laguerre(n, x, y), f.bernoulli(n), f.fibonacci(n, x), f.elliptic_f(x, y),
            f.elliptic_e(x), f.elliptic_pi(x, y, z), f.LambertW(x), f.LambertW(x, -1), f.re(x), f.im(x),
            f.conjugate(x), f.Max(x, y), f.Min(x, 1), f.erf(x), f.erf(x)**2, f.Ei(x), f.atan2(y, x),
            f.KroneckerDelta(x, y), f.DiracDelta(x), f.DiracDelta(x, 1), f.Heaviside(x), f.Heaviside(x, 0),
            f.hyper([x, y], [z], x), f.airyaiprime(x), f.fresnels(x), f.uppergamma(x, y), mod(x, y),
            mod(-x, y), f.sign(x), f.sinh(x)*f.cosh(x), Eq(x, y), Ne(x, 2), x < y, x >= y + 1, Not(Eq(x, y)),
            And(x < y, y < z), Or(x > 1, x < -1), Xor(x > 1, y > 1), Implies(x > 1, y > 1), Not(Implies(x > 1, y > 1)),
            Equivalent(x > 1, y > 1), F(x), F(x, y)**2, G(x), F(x)*F(y), Derivative(F(x), x),
            Derivative(F(x, y), x, y), Derivative(F(x), (x, 2)), Derivative(F(x), x)*F(x),
            Subs(Derivative(F(x), x), x, 1), f.exp(x)/x, x*f.sin(x)/f.cos(y), (x + y)/(x - y)**2]


@pytest.mark.parametrize("index", range(len(_cases())))
def test_latex(rust: Any, index: int) -> None:
    from sympy_extras.printing.latex import latex
    rc = load("sympy_extras_rs.core")
    e = _cases()[index]
    for settings in SETTINGS:
        got = _outcome(lambda: rc.latex(e, settings))
        if got[0] == 'SKIP':
            continue
        assert got == _outcome(lambda: latex(e, **settings)), (e, settings)


def _random(seed: int) -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, pi
    from sympy_extras import functions as f
    x, y, p = Symbol('x'), Symbol('y'), Symbol('p', positive=True)
    rng = random.Random(stable_seed("latex", seed))
    atoms = [x, y, p, Integer(2), Integer(-3), Rational(1, 2), Rational(-2, 3), pi, I]
    funcs = [f.sin, f.cos, f.exp, f.log, f.sqrt, lambda a: a**2, lambda a: 1/a, f.gamma, f.Abs, lambda a: a**Rational(1, 3),
             lambda a: -a, f.factorial]

    def rnd(d: int) -> Any:
        if d == 0 or rng.random() < .25:
            return rng.choice(atoms)
        r = rng.random()
        if r < .35:
            return rnd(d - 1) + rnd(d - 1)
        if r < .7:
            return rnd(d - 1)*rnd(d - 1)
        return rng.choice(funcs)(rnd(d - 1))
    out = []
    while len(out) < 30:
        try:
            out.append(rnd(4))
        except (ValueError, ZeroDivisionError):
            continue
    return out


@pytest.mark.parametrize("seed", range(6))
def test_latex_random(rust: Any, seed: int) -> None:
    from sympy_extras.printing.latex import latex
    rc = load("sympy_extras_rs.core")
    rng = random.Random(seed)
    different = []
    for e in _random(seed):
        settings = rng.choice(SETTINGS)
        got = _outcome(lambda: rc.latex(e, settings))
        if got[0] != 'SKIP' and got != _outcome(lambda: latex(e, **settings)):
            different.append((e, settings, got))
    assert not different, different[:3]
