"""Tests of the native Almkvist–Zeilberger algorithm on the objects of the
core (:mod:`sympy_extras.integrals.telescoping`)."""
from __future__ import annotations

from sympy_extras.core import Function, Integer, Rational, Symbol, diff, expand, oo, I
from sympy_extras.core.expr import Expr
from sympy_extras.functions import cos, cosh, exp, sin, sqrt
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.integrals.telescoping import (_Context, _nullspace, almkvist_zeilberger, holonomic_integral,
                                                holonomic_ode, is_hyperexponential, logarithmic_derivative)
from sympy_extras.polys.polytools import cancel
from sympy_extras.settings import configure
from sympy_extras.simplify.simplify import simplify

x = Symbol('x')
t = Symbol('t', positive=True)


def _telescopes(F: Expr) -> None:
    """The telescoper of ``F`` exists and satisfies its identity."""
    found = almkvist_zeilberger(F, x, t)
    assert found is not None, F
    assert found.check(), F
    lhs = sum((a*diff(F, (t, j)) for j, a in enumerate(found.coefficients)), Integer(0))
    # checked independently of DifferentialTelescoper.check: by the
    # exponential form
    difference = ((lhs - diff(found.certificate*F, x))/F).rewrite(Exp)
    assert simplify(cancel(expand(difference))) == 0, F


def test_telescopers_satisfy_their_identity() -> None:
    for F in [exp(-x**2 - t**2/x**2), 1/(x**2 + t**2), exp(-x**2)*exp(2*x*t), exp(-t*x)*x**3/(x**2 + t**2),
              x**2*exp(-t*x), exp(-x)*sin(t*x), exp(-x**2)*cosh(2*t*x)]:
        _telescopes(F)


def test_known_telescopers() -> None:
    found = almkvist_zeilberger(exp(-x**2 - t**2/x**2), x, t)
    assert found is not None and found.coefficients == [-4, 0, 1] and found.certificate == 2/x
    found = almkvist_zeilberger(1/(x**2 + t**2), x, t)
    assert found is not None and found.coefficients == [1, t] and found.certificate == -x
    # an integrand with a hyperexponential antiderivative: order zero
    found = almkvist_zeilberger(x*exp(-t*x**2), x, t)
    assert found is not None and found.order == 0
    assert almkvist_zeilberger(exp(exp(x))*t, x, t) is None
    # not hyperexponential in t: the logarithmic derivative log(x)
    assert almkvist_zeilberger(x**t*exp(-x), x, t) is None


def test_gaussian_coefficients() -> None:
    # the cosine written as exponentials: the polynomials have their
    # coefficients in QQ_I(t), and the terms share their telescoper
    found = almkvist_zeilberger(exp(-x**2)*cos(2*t*x), x, t)
    assert found is not None and found.coefficients == [2*t, 1] and found.check()
    u = Symbol('u')
    found = almkvist_zeilberger(exp(-x**2 + I*u*x), x, u)
    assert found is not None and found.check()
    # a transcendental constant is a generator of the coefficient field
    _telescopes(exp(-x**2)*exp(sqrt(2)*t*x))


def test_logarithmic_derivatives() -> None:
    assert logarithmic_derivative(x**t*exp(-x**2), x) == (t - 2*x**2)/x
    assert logarithmic_derivative(exp(exp(x)), x) is None
    assert logarithmic_derivative(Integer(0), x) is None
    assert is_hyperexponential(exp(-x*t)*x**3/(x**2 + t**2), x, t)
    assert not is_hyperexponential(exp(-x*t)*sin(x), x, t)


def test_equations_of_the_integral() -> None:
    I_ = Function('I')(t)
    equation = holonomic_ode(exp(-x**2)*exp(2*x*t), x, -oo, oo, t)
    assert equation is not None and equation.lhs == -2*t*I_ + diff(I_, t) and equation.rhs == 0
    # a boundary term which is not finite: no equation
    assert holonomic_ode(exp(x*t), x, 0, oo, t) is None


def test_integrals_of_order_zero() -> None:
    # the equation I(t) = [R F] needs neither initial values nor dsolve
    with configure(numerical_checks=False):
        found = holonomic_integral(x*exp(-t*x**2), x, 0, oo, t)
    assert found is not None and found.value == 1/(2*t)


def test_null_spaces_over_the_parameters() -> None:
    a = Symbol('a')
    basis = _nullspace([[a, Integer(1), Integer(0)], [Integer(0), a, Integer(1)]], 3)
    assert basis is not None and len(basis) == 1
    v = basis[0]
    assert v[2] == 1 and cancel(a*v[0] + v[1]) == 0 and cancel(a*v[1] + v[2]) == 0
    # Gaussian entries
    basis = _nullspace([[I, Integer(1)]], 2)
    assert basis == [[I, Integer(1)]]
    # an entry outside the rational functions
    assert _nullspace([[exp(a), Integer(1)]], 2) is not None
    assert _nullspace([[Rational(1, 2), x]], 2) is not None


def test_factorisation_over_the_field() -> None:
    # over QQ(t) x**2 + t**2 is irreducible; over QQ_I(t) it splits
    rational = _Context([x**2 + t**2], x)
    _, factors = rational.poly(x**2 + t**2).factor_list()
    assert len(factors) == 1
    gaussian = _Context([x**2 + t**2, I*x], x)
    _, factors = gaussian.poly(x**2 + t**2).factor_list()
    assert sorted(str(gaussian.poly_expr(f.monic())) for f, _ in factors) == ['-I*t + x', 'I*t + x']


def test_symbols_named_like_the_internal_variables() -> None:
    # the typed polynomials know their variables by name: the generators
    # are renamed, so a parameter called like an internal name is safe
    k = Symbol('_k0')
    kx = Symbol('_kx')
    _telescopes(exp(-x**2)*exp(2*x*t)*k)
    found = almkvist_zeilberger(1/(x**2 + kx**2), x, kx)
    assert found is not None and found.coefficients == [1, kx]
