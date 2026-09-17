"""Tests of Trager's algorithm for algebraic functions (quadratic extensions)."""
from __future__ import annotations

from sympy import symbols, sqrt, log, I, S, Pow, simplify, exp, sin
from sympy.polys.polyerrors import NotInvertible
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing import as_expr
from sympy_extras.integrals.trager import (trager_antiderivative, trager_reduce, hermite_reduce_algebraic,
                                           is_nonelementary_algebraic)

x, a = symbols('x a')


def _derivative_matches(F: object, f: object) -> bool:
    from sympy_extras._typing import as_expr
    return simplify(as_expr(F).diff(x) - as_expr(f)) == 0


def test_antiderivatives_are_verified_by_differentiation() -> None:
    cases = [
        (1 / (x * sqrt(x**2 + 1)), -log((sqrt(x**2 + 1) + 1) / x)),
        (sqrt(x**2 + 1) / x, sqrt(x**2 + 1) - log((sqrt(x**2 + 1) + 1) / x)),
        (x / sqrt(x**4 + 1), log(x**2 + sqrt(x**4 + 1)) / 2),
        (1 / sqrt(x**2 - 1), log(x + sqrt(x**2 - 1))),
        (x**2 / sqrt(x**2 + 1), x * sqrt(x**2 + 1) / 2 - log(x + sqrt(x**2 + 1)) / 2),
        (1 / (x * sqrt(x**3 + 1)), -log((x**3 / 2 + sqrt(x**3 + 1) + 1) / x**3) / 3),
        (1 / (x * sqrt(x**4 + 1)), -log((sqrt(x**4 + 1) + 1) / x**2) / 2),
    ]
    for f, expected in cases:
        found = trager_antiderivative(f, x)
        assert found is not None and found == expected, (f, found)
        assert _derivative_matches(found, f)
    # a class of residues in a number field: the logarithm of a function
    # with complex coefficients (a real form would be an arctangent)
    found = trager_antiderivative((x**2 - 1) / ((x**2 + 1) * sqrt(x**4 + 1)), x)
    assert found is not None and found.has(I) and _derivative_matches(found, (x**2 - 1) / ((x**2 + 1) * sqrt(x**4 + 1)))
    found = trager_antiderivative(1 / ((x**2 + 1) * sqrt(x**2 + 2)), x)
    assert found is not None and _derivative_matches(found, 1 / ((x**2 + 1) * sqrt(x**2 + 2)))


def test_hermite_reduction() -> None:
    # multiple poles removed, the polynomial part reduced at infinity
    found = hermite_reduce_algebraic(x**2 / sqrt(x**2 + 1), x)
    assert found == (x * sqrt(x**2 + 1) / 2, -1 / (2 * sqrt(x**2 + 1)))
    found = hermite_reduce_algebraic(1 / (x**2 * sqrt(x**3 + 1)), x)
    assert found == (-sqrt(x**3 + 1) / x, x / (2 * sqrt(x**3 + 1)))
    f = (x**3 + 1) / ((x - 1)**2 * sqrt(x**3 + 1))
    found = hermite_reduce_algebraic(f, x)
    assert found is not None
    g, h = found
    assert simplify(g.diff(x) + h - f) == 0
    # a ramified pole (a factor of the radicand in the denominator)
    f = 1 / (x * sqrt(x)) + 1 / sqrt(x)
    found = hermite_reduce_algebraic(f, x)
    assert found is not None and simplify(found[0].diff(x) + found[1] - f) == 0
    # a rational part is reduced too (Horowitz-Ostrogradsky)
    f = 1 / (x - 1)**2 + 1 / sqrt(x**2 + 1)
    found = hermite_reduce_algebraic(f, x)
    assert found is not None and simplify(found[0].diff(x) + found[1] - f) == 0 and found[0].has(1 / (x - 1))


def test_nonelementary_integrals_are_left_alone() -> None:
    # differentials of the first kind and poles of order two at infinity
    assert trager_antiderivative(1 / sqrt(x**3 + 1), x) is None
    assert trager_reduce(1 / sqrt(x**3 + 1), x) == (0, 1 / sqrt(x**3 + 1))
    assert trager_reduce(1 / sqrt(1 - x**4), x) == (0, 1 / sqrt(1 - x**4))
    assert is_nonelementary_algebraic(x / sqrt(x**3 + 1), x) is True
    assert is_nonelementary_algebraic(1 / (x**2 * sqrt(x**3 + 1)), x) is True
    assert is_nonelementary_algebraic(x / sqrt(x**4 + 1), x) is False
    # the elementary part split off, the rest a differential of the first kind
    f = (2 * x**2 + 1) / ((x**2 + 1) * sqrt(x**4 + x**2 + 1))
    found = trager_reduce(f, x)
    assert found is not None
    F, h = found
    assert h == 3 / (2 * sqrt(x**4 + x**2 + 1)) and simplify(F.diff(x) + h - f) == 0
    assert is_nonelementary_algebraic(f, x) is True
    # an irrational residue on a curve of genus one: not decided
    assert is_nonelementary_algebraic(1 / ((x - 1) * sqrt(x**3 + 1)), x) is None


def test_integrands_outside_the_scope() -> None:
    assert trager_antiderivative(exp(x) * sqrt(x), x) is None
    # two roots: the product of the radicands (test_several_roots_are_combined_and_restored)
    product = trager_antiderivative(sqrt(x) * sqrt(x + 1), x)
    assert product is not None and product.has(sqrt(x) * sqrt(x + 1))
    assert trager_antiderivative(sqrt(x) + sqrt(x + 1), x) is None
    assert trager_antiderivative(sin(x), x) is None
    assert hermite_reduce_algebraic(sqrt(sqrt(x) + 1), x) is None
    # a parameter: the Hermite reduction only
    found = hermite_reduce_algebraic(x**2 / sqrt(x**2 + a), x)
    assert found is not None and simplify(found[0].diff(x) + found[1] - x**2 / sqrt(x**2 + a)) == 0
    # the logarithmic part over QQ(a): log(x + sqrt(x**2 + a)), decided
    assert is_nonelementary_algebraic(1 / sqrt(x**2 + a), x) is False
    found_ = trager_antiderivative(1 / sqrt(x**2 + a), x)
    assert found_ is not None and simplify(found_.diff(x) - 1 / sqrt(x**2 + a)) == 0


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(trager_antiderivative)([1], x))
    raises(TypeError, lambda: untyped(trager_reduce)(None, x))


def test_perfect_square_radicands_carry_the_sign() -> None:
    # sqrt(d**2) is d*sign(d): the antiderivative of the rational
    # integrand on each component, and the verification ignores the jump
    from sympy import sign
    from sympy_extras.integrals import verified_antiderivative
    b, c = symbols('b c')
    assert trager_antiderivative(sqrt(x**2), x) == x**2 * sign(x) / 2
    assert trager_antiderivative(sqrt(x**2 + 2 * x + 1) / (x + 3), x) == (x - 2 * log(x + 3)) * sign(x + 1)
    f = as_expr((b**2 / (4 * c) + b * x + c * x**2)**(-S(3) / 2))
    found = verified_antiderivative(f, x)
    assert found is not None and found[0].has(sign(b / (2 * c) + x))


def test_several_roots_are_combined_and_restored() -> None:
    # sqrt(x + 1)*sqrt(x + 2) is the root of the product where both are
    # positive and its negative below -2: the algebra runs in the root of
    # the product, the answer is written in the original roots and holds
    # on both sides
    from sympy_extras.integrals import is_antiderivative, verified_antiderivative
    f = sqrt(x + 1) * sqrt(x + 2) / x
    F = trager_antiderivative(f, x)
    assert F is not None and F.has(sqrt(x + 1) * sqrt(x + 2)) and not F.has(sqrt(x**2 + 3 * x + 2))
    assert is_antiderivative(F, f, x) is True
    found = verified_antiderivative(f, x)
    assert found is not None and found[1] == 'trager'
    # a monomial with one root only is not rational in the product
    assert trager_antiderivative(sqrt(x + 1) * sqrt(x + 2) + sqrt(x + 1), x) is None
    # the product of the radicands of genus one: the elliptic remainder
    found_ = trager_reduce((x**2 - 1) / (x**2 * sqrt((1 - 3 * x**2) * (1 - x**2))), x)
    assert found_ is not None and found_[1] != 0


def test_the_logarithmic_part_over_the_parameters() -> None:
    # FriCAS's tests 301 and 19: residues rational in the parameter, the
    # field QQ(a) (the residue at infinity 1/k, the branch sqrt(k**2) taken as k)
    from sympy_extras.integrals import is_antiderivative
    k = symbols('k')
    f = as_expr((-a + x * (2 * a - 1)) / ((-a + x) * sqrt(a**2 * x + x**3 * (2 * a - 1) - x**2 * (a**2 + 2 * a - 1))))
    F = trager_antiderivative(f, x)
    assert F is not None and F.has(log) and is_antiderivative(F, f, x) is True
    g = as_expr(x / sqrt((1 - x**2) * (1 - k**2 * x**2)))
    G = trager_antiderivative(g, x)
    assert G is not None and is_antiderivative(G, g, x) is True


def test_residues_algebraic_over_the_parameters() -> None:
    # FriCAS's test 295: the residues are sqrt(b -+ 2*sqrt(a*c))/(4*sqrt(a*c))
    # and I times them; the parameters are made rational (a = alpha**2,
    # c = gamma**2, b = (beta1**2 + beta2**2)/2), the algorithm runs over
    # QQ<I>(alpha, beta1, beta2, ...), and the answer's radicand is the
    # original one
    from sympy_extras.integrals import is_antiderivative
    from sympy_extras.integrals.trager import _rational_parameters
    b, c = symbols('b c')
    f = as_expr(sqrt(a + b * x**2 + c * x**4) / (a - c * x**4))
    F = trager_antiderivative(f, x)
    assert F is not None and F.has(sqrt(a + b * x**2 + c * x**4)) and F.has(log)
    assert is_antiderivative(F, f, x) is True
    found = _rational_parameters([sqrt(b - 2 * sqrt(a) * sqrt(c)) / 4, sqrt(c)], [a, b, c])
    assert found is not None
    substitution, back = found
    assert set(substitution) <= {a, b, c} and all(not value.has(sqrt) for value in substitution.values())
    assert all(any(node.exp == S.Half for node in value.atoms(Pow)) for value in back.values())
    # a root linear in no parameter: no reparametrization
    assert _rational_parameters([sqrt(a**2 + 1)], [a]) is None


def test_rational_radicands() -> None:
    # sqrt(P/Q) is y/Q with y = Q*sqrt(P/Q), y**2 = P*Q everywhere; the
    # answer is written in the root as given
    from sympy_extras.integrals import is_antiderivative
    f = as_expr(sqrt((x + 1) / x) / x)
    F = trager_antiderivative(f, x)
    assert F is not None and F.has(sqrt((x + 1) / x)) and not F.has(sqrt(x**2 + x))
    assert is_antiderivative(F, f, x) is True
    assert is_antiderivative(F, f, x, [x < -1]) is True
    # FriCAS's test 51, genus one after x**3 is cleared: the remainder stays
    found = trager_reduce((x + 1) / (x**3 * sqrt((-x**3 + 9 * x**2 + 12 * x + 4) / x**3)), x)
    assert found is not None and found[1] != 0


def test_roots_of_index_three_and_more() -> None:
    # y = P**(1/n): the components y**k integrate on their own through the
    # Risch differential equation R' + k*P'/(n*P)*R = A_k
    from sympy import Rational
    from sympy_extras.integrals import is_antiderivative, verified_antiderivative
    from sympy_extras.integrals.trager import _root_components
    cases = [x**2 * (x**3 + 1)**Rational(1, 3), x**5 * (x**3 + 1)**Rational(2, 3), x**2 * (x**3 + a)**Rational(1, 3),
             x / (x**2 + 1)**Rational(1, 3), (x**3 + 1)**Rational(-2, 3) * x**2 + (x**3 + 1)**Rational(1, 3) * x**2]
    for f in cases:
        F = trager_antiderivative(f, x)
        assert F is not None and is_antiderivative(F, f, x) is True, f
    found = verified_antiderivative(as_expr(x**2 * (x**3 + a)**Rational(1, 3)), x)
    assert found is not None and found[1] == 'trager'
    # no rational solution: undecided (the binomial substitution integrates it)
    g = as_expr(1 / (x * (x**3 + 1)**Rational(1, 3)))
    assert trager_antiderivative(g, x) is None
    reduced = trager_reduce(g, x)
    assert reduced is not None and reduced[0] == 0 and simplify(reduced[1] - g) == 0
    found = verified_antiderivative(g, x)
    assert found is not None and found[1] == 'rewriting'
    # the components: 1/(1 + y) is (1 - y + y**2)/(1 + P) for y**3 = P
    parsed = _root_components(as_expr(1 / (1 + (x**3 + 1)**Rational(1, 3))), x)
    assert parsed is not None
    P, n, components = parsed
    assert n == 3 and [simplify(c * (x**3 + 2)) for c in components] == [1, -1, 1]
    # not squarefree, a square root, two radicands: not this route
    assert _root_components(as_expr((x**2 * (x + 1))**Rational(1, 3)), x) is None
    assert _root_components(as_expr(sqrt(x**3 + 1)), x) is None
    assert _root_components(as_expr((x + 1)**Rational(1, 3) * (x + 2)**Rational(1, 3)), x) is None


def test_the_inverse_modulo_a_polynomial_over_the_gaussian_parametric_field() -> None:
    # Poly.invert reports a zero divisor over QQ<I>(a, b) where the
    # extended Euclidean algorithm finds the unit
    from sympy import Poly, QQ, cancel
    from sympy_extras.integrals.trager import _inverse
    b = symbols('b')
    K = QQ.algebraic_field(I).frac_field(a, b)
    G = Poly(x**2 - (a**2 + b**2) / 4, x, domain=K)
    p = Poly(4 * x**3, x, domain=K).mul_ground(K.from_sympy(I * b / (a**2 + b**2)))
    inverse = _inverse(p, G)
    assert cancel((p * inverse).rem(G).as_expr()) == 1
    raises(NotInvertible, lambda: _inverse(Poly(x, x, domain=K), Poly(x**2, x, domain=K)))
