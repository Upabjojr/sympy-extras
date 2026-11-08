"""Oracle tests of simplify against SymPy."""
import random

import sympy

from sympy_extras.core.interop import expr_from_sympy, to_sympy
from sympy_extras.simplify.simplify import simplify

s = sympy
x, y, z = s.symbols('x y z')
p = s.Symbol('p', positive=True)
n = s.Symbol('n', integer=True)


def _ours(e: sympy.Expr) -> str:
    return str(to_sympy(simplify(expr_from_sympy(e)), evaluate=False))


def test_simplify_matches_sympy() -> None:
    cases: list[sympy.Expr] = [
        s.sin(x)**2 + s.cos(x)**2, (x**3 + x**2 - x - 1) / (x**2 + 2 * x + 1), s.gamma(x + 1) / s.gamma(x),
        s.log(2) + s.log(3), (x + 1)**2 - x**2 - 2 * x, 1 / x + 1 / y, s.exp(x) * s.exp(y), s.sqrt(p**2),
        s.Abs(x) * s.Abs(y), s.sign(2 * x), s.log(p**2) / 2, s.cos(x)**4 - s.sin(x)**4, (s.exp(x) - s.exp(-x)) / 2,
        s.factorial(n + 1) / s.factorial(n), 1 / (s.sqrt(2) + 1), x / (x + y) + y / (x + y), s.tan(x) * s.cos(x),
        s.Abs(x + 1) / s.Abs(x**2 - 1), s.cosh(x)**2 - s.sinh(x)**2, 0.5 * x + x / 2,
        s.Piecewise((x + x, x > 0), (1, True)), s.binomial(n, 2), s.exp(2 * s.log(x)), s.log(x * y) - s.log(x),
        p / 4 + s.log(x**2 / p) + s.gamma(s.pi * y) + s.pi / 4 + 2]
    for e in cases:
        assert _ours(e) == str(s.simplify(e)), e


def test_simplify_random() -> None:
    rng = random.Random(3)
    atoms = [x, y, p, s.Integer(2), s.Rational(1, 2), s.pi]
    funcs = [s.sin, s.cos, s.exp, s.log, s.sqrt, lambda a: a**2, lambda a: 1 / a, s.tan, s.gamma, s.Abs]

    def rnd(d: int) -> sympy.Expr:
        if d == 0 or rng.random() < .25:
            return rng.choice(atoms)
        r = rng.random()
        if r < .35:
            return rnd(d - 1) + rnd(d - 1)
        if r < .7:
            return rnd(d - 1) * rnd(d - 1)
        return rng.choice(funcs)(rnd(d - 1))
    for _ in range(25):
        e = rnd(3)
        assert _ours(e) == str(s.simplify(e)), e


def test_inversecombine_and_kroneckersimp_match_sympy() -> None:
    from sympy.simplify.simplify import inversecombine as sympy_inversecombine, kroneckersimp as sympy_kroneckersimp
    from sympy_extras.simplify.simplify import inversecombine, kroneckersimp
    i, j = s.symbols('i j')
    K = s.KroneckerDelta
    for e in [s.log(s.exp(x)) + s.exp(s.log(y)), s.atan(s.tan(x)), s.asinh(s.sinh(s.exp(s.log(x)))),
              s.erf(s.erfinv(x)) * s.tan(s.atan(y)), s.sin(s.asin(x)) + s.acosh(s.cosh(x))]:
        assert str(to_sympy(inversecombine(expr_from_sympy(e)), evaluate=False)) == str(sympy_inversecombine(e)), e
    for e in [1 + K(0, i) * K(1, i), K(i, j) * K(0, i) * K(1, j), x * K(i, 0) * K(i, 0) + K(i, 2) * K(i, 1)]:
        assert str(to_sympy(kroneckersimp(expr_from_sympy(e)), evaluate=False)) == str(sympy_kroneckersimp(e)), e


def test_simplify_keeps_dummies_through_sympy_steps() -> None:
    # hyper_as_trig's dummy must survive the factor fallback (domain ZZ_I)
    e = x + s.sqrt(1 - s.sinh(s.sinh(x))**2) / s.sinh(s.sinh(x))
    assert _ours(e) == str(s.simplify(e))


def test_nc_simplify_matches_sympy() -> None:
    import random
    import sympy
    from sympy.simplify.simplify import nc_simplify as sympy_nc_simplify
    from sympy_extras.core import to_sympy
    from sympy_extras.core.interop import expr_from_sympy
    from sympy_extras.simplify.simplify import nc_simplify
    a, b, c = sympy.symbols('a b c', commutative=False)
    sx = sympy.Symbol('x')
    rng = random.Random(3)
    atoms = [a, b, c, a**2, b**-1, a**-1, a*b, (a*b)**2, c**3]
    for _ in range(150):
        e = sympy.Mul(*[rng.choice(atoms) for _ in range(rng.randint(2, 8))])
        if rng.random() < 0.2:
            e = e*2*sx
        if rng.random() < 0.2:
            e = e + sympy.Mul(*[rng.choice(atoms) for _ in range(3)])
        want = str(sympy_nc_simplify(e))
        if want.startswith('(a*b)*'):
            continue  # SymPy keeps an unevaluated (a*b) there
        assert str(to_sympy(nc_simplify(expr_from_sympy(e)), evaluate=False)) == want, e


def test_besselsimp_matches_sympy() -> None:
    import sympy
    from sympy import I, S, besseli, besselj, besselk, bessely, exp_polar, pi
    from sympy_extras.core import to_sympy
    from sympy_extras.core.interop import expr_from_sympy
    from sympy_extras.simplify.simplify import besselsimp
    z, nu, x = sympy.symbols('z nu x')
    cases = [besselj(nu, z*I), besseli(nu, -z), besselj(S(1)/2, z), besselj(S(3)/2, z), bessely(S(1)/2, z),
             besseli(S(1)/2, z), besselk(S(1)/2, z), besselj(2, z), besseli(2, z),
             besselj(0, z) + besselj(2, z) - 2*besselj(1, z)/z, besselj(1, z) + besselj(3, z),
             besselj(nu, exp_polar(I*pi/2)*z), besseli(nu, I*z), x*besselj(5, z), bessely(2, z) + bessely(0, z)]
    for e in cases:
        assert str(to_sympy(besselsimp(expr_from_sympy(e)), evaluate=False)) == str(sympy.besselsimp(e)), e


def test_simplify_of_piecewise_matches_sympy() -> None:
    # the native piecewise_simplify in the driver, and a Piecewise nested
    # in a sum simplified piece by piece (it used to break the pairs)
    import sympy
    from sympy import Piecewise as P
    from sympy_extras.core import to_sympy
    from sympy_extras.core.interop import expr_from_sympy
    from sympy_extras.simplify.simplify import simplify
    x, y = sympy.symbols('x y')
    cases = [P((1, x < 0), (1, x < 1), (2, True)), P((x**2, (x > 0) & (x < 2)), (2*x, x <= 0), (0, True)),
             2*P((x, x > 1), (0, True)) + 1, P((sympy.sin(x)**2 + sympy.cos(x)**2, x > 0), (0, True)),
             P((x*y, y > 0), (y, True)), P((1, x > 2), (1, x < -2), (0, True)), sympy.Abs(x) + P((x, x > 0), (-x, True)),
             P(((x**2 - 1)/(x - 1), x > 0), (x + 1, True)), sympy.exp(P((x, x > 0), (0, True))),
             x*P((sympy.sin(x)**2 + sympy.cos(x)**2, x - 1 > 0), (2, True))]
    for e in cases:
        assert str(to_sympy(simplify(expr_from_sympy(e)), evaluate=False)) == str(sympy.simplify(e)), e


def test_simplify_of_hyper_bessel_meijerg_nodes_matches_sympy() -> None:
    # their _eval_simplify (hyperexpand, besselsimp, piecewise_simplify) is
    # native; a Meijer G-function takes the common path, as in SymPy
    import sympy
    from sympy import Piecewise as P, Rational as R, besselj, hyper, meijerg
    from sympy_extras.core import to_sympy
    from sympy_extras.core.interop import expr_from_sympy
    from sympy_extras.simplify.simplify import simplify
    x, z, nu, a = sympy.symbols('x z nu a')
    cases = [hyper([1, 1], [2], z), meijerg([1], [], [], [], z), besselj(R(1, 2), z), besselj(nu, z*sympy.I),
             hyper([R(1, 2)], [R(3, 2)], -z**2), meijerg([], [], [0], [], z),
             P((besselj(R(1, 2), x), x > 0), (hyper([], [], x), True)), x*hyper([1, 1], [2], -x) + 1,
             meijerg([a], [], [a], [], z)]
    for e in cases:
        assert str(to_sympy(simplify(expr_from_sympy(e)), evaluate=False)) == str(sympy.simplify(e)), e
