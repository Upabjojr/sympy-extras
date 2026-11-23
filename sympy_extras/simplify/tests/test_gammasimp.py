"""Oracle tests of gammasimp and combsimp against SymPy."""
import random

import sympy

from sympy_extras.interop import expr_from_sympy, to_sympy
from sympy_extras.simplify.gammasimp import combsimp, gammasimp

s = sympy
x, y = s.symbols('x y')
n, k = s.symbols('n k', integer=True)


def _ours(f: object, e: sympy.Expr) -> str:
    assert f is gammasimp or f is combsimp
    result = gammasimp(expr_from_sympy(e)) if f is gammasimp else combsimp(expr_from_sympy(e))
    return str(to_sympy(result, evaluate=False))


def test_gammasimp_identities() -> None:
    cases: list[sympy.Expr] = [
        s.gamma(x) / s.gamma(x - 3), s.gamma(x) * s.gamma(1 - x), s.gamma(x) * s.gamma(x + s.S.Half) / s.gamma(2 * x),
        s.gamma(x) * s.gamma(x + s.Rational(1, 3)) * s.gamma(x + s.Rational(2, 3)), s.gamma(s.Rational(7, 3)),
        x * s.gamma(x) / s.gamma(x + 1), s.beta(x, y), s.rf(x, 3) * s.gamma(x), s.gamma(x) * s.gamma(-x),
        s.exp(s.gamma(x + 1) / s.gamma(x)), (s.gamma(x + 2) - s.gamma(x)) / s.gamma(x)]
    for e in cases:
        assert _ours(gammasimp, e) == str(s.gammasimp(e)), e


def test_combsimp_identities() -> None:
    cases: list[sympy.Expr] = [
        s.factorial(n) / s.factorial(n - 3), s.binomial(n + 1, k + 1) / s.binomial(n, k),
        s.factorial(n + k) / (s.factorial(n) * s.factorial(k)), s.rf(n, 3) / s.factorial(n + 2),
        s.ff(n, k) / s.factorial(n), s.factorial(2 * n) / s.factorial(n)**2, s.binomial(n, k) / s.binomial(n, k - 1),
        s.gamma(x + 1) / s.gamma(x)]
    for e in cases:
        assert _ours(combsimp, e) == str(s.combsimp(e)), e


def test_gammasimp_random_products() -> None:
    rng = random.Random(1)
    shifts = [-2, -1, 0, 1, 2, s.S.Half, -s.S.Half, s.Rational(1, 3)]
    for _ in range(15):
        e: sympy.Expr = s.S.One
        for _ in range(rng.randint(2, 4)):
            e *= s.gamma(rng.choice([x, y, 2 * x, x + y]) + rng.choice(shifts))**rng.choice([1, -1, 2])
        assert _ours(gammasimp, e) == str(s.gammasimp(e)), e


def test_gammasimp_fractional_shift_differing_by_integers() -> None:
    # issue #25: SymPy leaves gamma(n + 3/2)/gamma(n + 1/2); it is n + 1/2
    from sympy_extras.core import Rational, Symbol
    from sympy_extras.functions import gamma
    m, z = Symbol('m', integer=True, nonnegative=True), Symbol('z')
    assert str(gammasimp(gamma(m + Rational(3, 2))/gamma(m + Rational(1, 2)))) == '(2*m + 1)/2'
    assert str(gammasimp(gamma(m + Rational(5, 4))/gamma(m + Rational(1, 4)))) == '(4*m + 1)/4'
    assert str(combsimp(gamma(z + Rational(7, 2))/gamma(z - Rational(3, 2)))) == \
        '(2*z - 3)*(2*z - 1)*(2*z + 1)*(2*z + 3)*(2*z + 5)/32'
    # a single gamma function is not rewritten
    assert gammasimp(gamma(z - Rational(1, 2))) == gamma(z - Rational(1, 2))


def test_rising_factorial_of_a_general_k_has_no_reflected_branch() -> None:
    # sympy/sympy#31275, V230: (-1)**k*gamma(1 - x)/gamma(1 - x - k) is rf(x, k) at the integers k
    # only; SymPy's rewrite gives it as the branch x <= 0 for every k
    import mpmath
    from sympy_extras.core import Rational, Symbol
    from sympy_extras.core.evalf import to_complex
    from sympy_extras.functions import RisingFactorial, FallingFactorial, binomial, ff, gamma, rf
    from sympy_extras.functions.elementary.piecewise import Piecewise
    from sympy_extras.functions.special.gamma_functions import Gamma
    z, n, k = Symbol('x'), Symbol('n'), Symbol('k')
    result = gammasimp(rf(z + n, k)*binomial(n, k))
    assert not result.has(Piecewise)
    assert result == gamma(n + 1)*gamma(k + n + z)/(gamma(k + 1)*gamma(n + z)*gamma(-k + n + 1))
    values = {z: Rational(-33, 10), n: Rational(11, 10), k: Rational(1, 2)}
    expected = mpmath.rf(-2.2, 0.5)*mpmath.binomial(1.1, 0.5)   # -1.5067...
    assert abs(to_complex(result.subs(values)) - complex(expected)) < 1e-12
    assert gammasimp(ff(z, k)) == gamma(z + 1)/gamma(z - k + 1)
    assert RisingFactorial(z, k).rewrite(Gamma) == gamma(z + k)/gamma(z)
    assert FallingFactorial(z, k).rewrite(Gamma) == gamma(z + 1)/gamma(z - k + 1)
    # at an integer k the two branches stay
    j = Symbol('j', integer=True)
    assert RisingFactorial(z, j).rewrite(Gamma).has(Piecewise) and FallingFactorial(z, j).rewrite(Gamma).has(Piecewise)
