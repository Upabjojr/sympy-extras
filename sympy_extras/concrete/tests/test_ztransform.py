"""The Z-transform against the defining series (summed numerically by
mpmath) and its inverse against the Taylor coefficients in 1/z computed
by SymPy."""
from __future__ import annotations

import mpmath
import pytest
import sympy

from sympy_extras.concrete.ztransform import inverse_z_transform, z_transform
from sympy_extras.core import Integer, Rational, Symbol, pi
from sympy_extras.core.expr import Expr
from sympy_extras.functions import KroneckerDelta, cos, exp, factorial, log, sin, sqrt
from sympy_extras.interop import sympy_expr, to_sympy

n, z, a, b, w = Symbol('n'), Symbol('z'), Symbol('a'), Symbol('b'), Symbol('w')
sn, sz, sa, sb, sw = sympy.symbols('n z a b w')
_VALUES = {sa: sympy.Rational(3, 5), sb: sympy.Rational(-4, 7), sw: sympy.Rational(11, 10)}


def _series_matches(f: Expr, image: Expr, point: float = 3.7) -> bool:
    """The image at a point outside the poles is the sum of the series."""
    sequence = sympy.lambdify(sn, sympy_expr(to_sympy(f)).subs(_VALUES), 'mpmath')
    expected = mpmath.nsum(lambda k: sequence(int(k))*mpmath.mpf(point)**(-int(k)), [0, mpmath.inf])
    found = sympy.N(sympy_expr(to_sympy(image)).subs(_VALUES).subs(sz, sympy.Float(point, 30)), 20)
    return bool(abs(complex(found) - complex(expected)) < 1e-10)


def test_forward_against_the_series() -> None:
    sequences: list[Expr] = [
        Integer(1), n, n**3, a**n, n**2*a**n, 2**(n + 1), exp(-n), exp(a*n + 1),
        cos(w*n), sin(w*n), a**n*cos(w*n + 1), n*sin(w*n), n**2*a**n*cos(n), sin(n)*cos(2*n), sin(n)**2, cos(n)**3,
        a**n/factorial(n), n*a**n/factorial(n), 1/(n + 1), a**n/(n + 3), n/(n + 2),
        3*KroneckerDelta(n, 0) - KroneckerDelta(n, 4)*n**2, (n + 1)**2*b**n - 5*a**n + cos(pi*n/3),
    ]
    for f in sequences:
        assert _series_matches(f, z_transform(f, n, z)), f


def test_forward_table() -> None:
    assert z_transform(1, n, z) == z/(z - 1)
    assert z_transform(n, n, z) == z/(z**2 - 2*z + 1)
    assert z_transform(a**n, n, z) == z/(z - a)
    assert z_transform(KroneckerDelta(n, 3), n, z) == z**-3
    assert z_transform(a**n/factorial(n), n, z) == exp(a/z)
    assert z_transform(1/(n + 1), n, z) == -z*log(1 - 1/z)
    assert z_transform(cos(pi*n/2), n, z) == z**2/(z**2 + 1)
    assert z_transform(0, n, z) == 0
    for unknown in (sqrt(n), 1/n, a**(n**2), factorial(n), cos(n**2), 1/(factorial(n)*(n + 1))):
        with pytest.raises(NotImplementedError):
            z_transform(unknown, n, z)


def _coefficients_match(image: Expr, sequence: Expr, terms: int = 9) -> bool:
    """The first Taylor coefficients in x = 1/z are the first terms."""
    x = sympy.Symbol('x')
    function = sympy_expr(to_sympy(image)).subs(sz, 1/x)
    expansion = sympy.series(sympy.cancel(function), x, 0, terms).removeO()
    found = sympy_expr(to_sympy(sequence))
    for k in range(terms):
        difference = (expansion.coeff(x, k) - found.subs(sn, k)).subs(_VALUES)
        if abs(complex(sympy.N(difference, 30))) > 1e-18:
            return False
    return True


def test_inverse_against_the_expansion() -> None:
    images: list[Expr] = [
        z/(z - 1), z/(z - a), 1/(z - a), z**2/((z - a)*(z - b)), z/(z - 2)**3, (z**2 + 1)/(z**2*(z - 3)),
        z/(z**2 - z - 1), z/(z**2 + 1), z**2/(z**2 - z + 1), (z + 2)/(z**2 + 2*z + 5), z**3/(z**2 + 2*z + 5)**2,
        z/(z**3 - 2), (3*z**2 - z)/((z - 1)**2*(z + Rational(1, 2))), z/((z - a)**2*(z - 1)), 1/z**3 + 2,
        z**2/(z**2 - a), z/(z**4 + 1), (z**3 + z)/((z**2 - 2)**2), z*(z - a*b)/(z**2 - (a + b)*z + a*b),
        z**4/((z**2 + z + 1)*(z**2 + 1)*(z - Rational(1, 3))),
    ]
    for image in images:
        assert _coefficients_match(image, inverse_z_transform(image, z, n)), image


def test_inverse_closed_forms() -> None:
    assert inverse_z_transform(z/(z - a), z, n) == a**n
    assert inverse_z_transform(z/((z - 1)*(z - 2)), z, n) == 2**n - 1
    assert inverse_z_transform(z*(z + 1)/(z - 1)**3, z, n) == n**2
    assert inverse_z_transform(z**2/(z**2 + 1), z, n) == cos(pi*n/2)
    assert inverse_z_transform(z/(z**2 - z + 1), z, n) == 2*sqrt(3)*sin(pi*n/3)/3
    assert inverse_z_transform(1/z**2, z, n) == KroneckerDelta(n, 2)
    assert inverse_z_transform(Integer(0), z, n) == 0 and inverse_z_transform(Integer(5), z, n) == 5*KroneckerDelta(n, 0)
    with pytest.raises(ValueError):
        inverse_z_transform(z**2/(z - 1), z, n)
    with pytest.raises(NotImplementedError):
        inverse_z_transform(exp(1/z), z, n)
    with pytest.raises(NotImplementedError):
        inverse_z_transform(z/(z**5 - z - 1), z, n)
    # coefficients which are not rational functions of symbols
    with pytest.raises(NotImplementedError):
        inverse_z_transform(z/(z**2 - sqrt(2)*z + 1), z, n)


def test_round_trips() -> None:
    for f in (n**2*3**n, 2**n - n, (n + 1)*Rational(1, 2)**n, cos(pi*n/3), 2**n*sin(pi*n/2) + 1, n**3):
        image = z_transform(f, n, z)
        back = inverse_z_transform(image, z, n)
        sympy_difference = sympy_expr(to_sympy(back - f))
        assert all(abs(complex(sympy.N(sympy_difference.subs(sn, k), 30))) < 1e-18 for k in range(10)), f
    for image in (z/(z - 3)**2, z**2/(z**2 + 4), (z**2 - z)/(z**2 - z + 1)):
        sequence = inverse_z_transform(image, z, n)
        again = z_transform(sequence, n, z)
        assert sympy.simplify(sympy_expr(to_sympy(again - image))) == 0, image
