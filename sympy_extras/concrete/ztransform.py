r"""The Z-transform of sequences and its inverse.

The (unilateral) Z-transform of a sequence $f(n)$, $n \ge 0$, is

.. math:: F(z) = \sum_{n=0}^{\infty} f(n)\, z^{-n},

the discrete counterpart of the Laplace transform [1]_.

**Forward.** The sequences $n^k a^n$, $n^k a^n \cos(\omega n + \varphi)$
and $n^k a^n \sin(\omega n + \varphi)$ and their linear combinations have
rational transforms, obtained from the geometric series

.. math:: a^n \mapsto \frac{z}{z - a}, \qquad
          a^n \cos \omega n \mapsto \frac{z (z - a \cos\omega)}{z^2 - 2 a z \cos\omega + a^2}, \qquad
          a^n \sin \omega n \mapsto \frac{a z \sin\omega}{z^2 - 2 a z \cos\omega + a^2}

by the rule $n f(n) \mapsto -z F'(z)$; products of sines and cosines are
linearised first. Three more families are known: $a^n / n!
\mapsto e^{a/z}$, $a^n / (n + m) \mapsto (z/a)^m \bigl(-\log(1 - a/z) -
\sum_{j=1}^{m-1} (a/z)^j / j\bigr)$ for an integer $m \ge 1$, and the
impulses $\delta_{n,m} \mapsto z^{-m}$. A step $H(n - m)$ ($m \ge 0$ an
integer) delays: $g(n) H(n - m) \mapsto z^{-m} \bigl(G_m(z) - (1 - H(0))
g(m)\bigr)$ with $G_m$ the transform of $g(n + m)$; ``Heaviside(n - m)``
has $H(0) = 1/2$, and ``Heaviside(n - m, 1)`` is the unit step of signal
processing.

**Inverse.** A rational $F(z)$ which is finite at infinity is a rational
function $A(x)/B(x)$ of $x = 1/z$ with $B(0) \ne 0$, and $f(n)$ is the
coefficient of $x^n$ in its power series [2]_. The polynomial part gives
impulses. The rest is split into partial fractions over the irreducible
factors $g$ of $B$ (over the field $K$ of rational functions in the
parameters); for each $g$ the expansion at a root $\rho$,

.. math:: \frac{R(x)}{g(x)^m} = \sum_{j=1}^{m} \frac{c_j(\rho)}{(x - \rho)^j},
          \qquad [x^n]\, \frac{1}{(x - \rho)^j} = (-\rho)^{-j} \binom{n + j - 1}{j - 1} \rho^{-n},

is computed once in the field $K[t]/(g)$ where $t$ is that root
(:mod:`sympy_extras.polys.domains.quotientfield`), and the roots are
substituted at the end; a quadratic factor with rational coefficients and
complex roots is written with $\cos$ and $\sin$.

Examples
========

>>> from sympy_extras.core import Symbol, pi
>>> from sympy_extras.functions import cos, factorial
>>> from sympy_extras.concrete.ztransform import z_transform, inverse_z_transform
>>> n, z, a = Symbol('n'), Symbol('z'), Symbol('a')
>>> z_transform(a**n, n, z), z_transform(n*2**n, n, z)
(z/(-a + z), 2*z/(z**2 - 4*z + 4))
>>> z_transform(cos(pi*n/2), n, z), z_transform(a**n/factorial(n), n, z)
(z**2/(z**2 + 1), exp(a/z))
>>> inverse_z_transform(z/(z - a), z, n), inverse_z_transform(z/(z - 2)**2, z, n)
(a**n, 2**n*n/2)
>>> inverse_z_transform(z/(z**2 - z - 1), z, n)       # the Fibonacci numbers
-sqrt(5)*(1/2 - sqrt(5)/2)**n/5 + sqrt(5)*(1/2 + sqrt(5)/2)**n/5
>>> inverse_z_transform(z**2/(z**2 + 1), z, n), inverse_z_transform(1/(z - 1), z, n)
(cos(pi*n/2), 1 - KroneckerDelta(0, n))

References
==========

.. [1] E. I. Jury, *Theory and Application of the z-Transform Method*,
       Wiley, 1964, Chapter 1 and the table of Appendix I.
.. [2] R. P. Stanley, *Enumerative Combinatorics*, vol. 1, 2nd ed.,
       Cambridge University Press, 2012, Theorem 4.1.1 (rational
       generating functions).
.. [3] A. V. Oppenheim, R. W. Schafer, *Discrete-Time Signal Processing*,
       3rd ed., Pearson, 2010, Chapter 3.
"""
from __future__ import annotations

from typing import Final, Optional, Sequence, TypeVar

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, Rational, number_from_fraction
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions import KroneckerDelta, acos, cos, exp, log, sin
from sympy_extras.functions.combinatorial.factorials import Factorial
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.functions.elementary.trigonometric import Cos, Sin
from sympy_extras.functions.special.delta_functions import Heaviside
from sympy_extras.functions.special.tensor_functions import KroneckerDeltaNode
from sympy_extras.matrices.eigen import _Field, _FunctionField, _RationalField
from sympy_extras.matrices.linalg import _PolynomialLift, _kernels, normal
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys import densearith, densebasic, euclidtools
from sympy_extras.polys.domains.quotientfield import QuotientField
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, factor_list
from sympy_extras.polys.rationaltools import together

__all__ = ['z_transform', 'inverse_z_transform']

T = TypeVar('T')

_ONE: Final[Expr] = Integer(1)
_ZERO: Final[Expr] = Integer(0)


# ---------------------------------------------------------------------
# forward

def _linear(e: Expr, n: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(alpha, beta)`` with ``e = alpha*n + beta``, or ``None``."""
    slope = e.diff(n)
    if slope.has(n):
        return None
    constant = e.subs(n, _ZERO)
    if expand(e - slope*n - constant) != 0:
        return None
    return slope, constant


class _Term:
    """``coefficient * n**power * base**n * trig`` with at most one of the
    special factors (``1/n!``, ``1/(n + shift)``, an impulse at
    ``impulse``)."""

    def __init__(self) -> None:
        self.coefficient: Expr = _ONE
        self.power: int = 0
        self.base: Expr = _ONE
        self.trig: list[Expr] = []
        self.factorial: bool = False
        self.shift: Optional[int] = None
        self.impulse: Optional[int] = None
        self.steps: list[Expr] = []


def _nonnegative_integer(e: Expr) -> Optional[int]:
    return e.p if isinstance(e, Integer) and e.p >= 0 else None


def _classify(term: Expr, n: Symbol) -> Optional[_Term]:
    result = _Term()
    for factor in (term.args if isinstance(term, Mul) else (term,)):
        if not isinstance(factor, Expr):
            return None
        if not factor.has(n):
            result.coefficient = result.coefficient*factor
        elif factor == n:
            result.power += 1
        elif isinstance(factor, Exp):
            parts = _linear(factor.exp, n)
            if parts is None:
                return None
            result.base = result.base*exp(parts[0])
            result.coefficient = result.coefficient*exp(parts[1])
        elif isinstance(factor, (Cos, Sin)):
            if _linear(factor.args[0], n) is None:
                return None
            result.trig.append(factor)
        elif isinstance(factor, Heaviside):
            argument = factor.args[0]
            parts = _linear(argument, n) if isinstance(argument, Expr) else None
            if parts is None or parts[0] != 1 or not isinstance(parts[1], Integer):
                return None
            result.steps.append(factor)
        elif isinstance(factor, KroneckerDeltaNode):
            first, second = factor.args
            position = _nonnegative_integer(first) if second == n else (_nonnegative_integer(second) if first == n else None)
            if position is None or result.impulse is not None:
                return None
            result.impulse = position
        elif isinstance(factor, Pow):
            base, exponent = factor.base, factor.exp
            if base == n:
                k = _nonnegative_integer(exponent)
                if k is None:
                    return None
                result.power += k
            elif not base.has(n):
                parts = _linear(exponent, n)
                if parts is None:
                    return None
                result.base = result.base*pow(base, parts[0])
                result.coefficient = result.coefficient*pow(base, parts[1])
            elif exponent == -1 and isinstance(base, Factorial) and base.args[0] == n and not result.factorial:
                result.factorial = True
            elif exponent == -1 and result.shift is None:
                parts = _linear(base, n)
                if parts is None or parts[0] != 1:
                    return None
                m = _nonnegative_integer(parts[1])
                if m is None or m == 0:
                    return None
                result.shift = m
            elif isinstance(base, (Cos, Sin)):
                k = _nonnegative_integer(exponent)
                if k is None or _linear(base.args[0], n) is None:
                    return None
                result.trig.extend([base]*k)
            else:
                return None
        else:
            return None
    return result


def _linearise(first: Expr, second: Expr) -> Expr:
    """The product of two sines or cosines as a sum of two."""
    u, v = first.args[0], second.args[0]
    if not isinstance(u, Expr) or not isinstance(v, Expr):
        raise TypeError('an argument which is not an expression')
    half = Rational(1, 2)
    if isinstance(first, Cos) and isinstance(second, Cos):
        return half*(cos(u - v) + cos(u + v))
    if isinstance(first, Sin) and isinstance(second, Sin):
        return half*(cos(u - v) - cos(u + v))
    if isinstance(first, Sin):
        return half*(sin(u + v) + sin(u - v))
    return half*(sin(u + v) - sin(u - v))


def _transform_term(term: Expr, n: Symbol, z: Symbol) -> Expr:
    data = _classify(term, n)
    if data is None:
        raise NotImplementedError('the Z-transform of %s' % (term,))
    if data.steps:
        # g(n) H(n - m): the terms from n = m on, the one at n = m weighted by H(0)
        step = data.steps[0]
        argument, at_zero = step.args[0], step.args[1] if len(step.args) > 1 else Rational(1, 2)
        if not isinstance(argument, Expr) or not isinstance(at_zero, Expr):
            raise NotImplementedError('the Z-transform of %s' % (term,))
        start = -argument.subs(n, _ZERO)
        if not isinstance(start, Integer):
            raise NotImplementedError('the Z-transform of %s' % (term,))
        rest = term/step
        if start.p < 0:
            return z_transform(rest, n, z)
        first = rest.subs(n, start)
        return pow(z, -start)*(z_transform(rest.subs(n, n + start), n, z) - (1 - at_zero)*first)
    if len(data.trig) > 1:
        rest = term
        for factor in data.trig[:2]:
            rest = rest/factor
        return z_transform(expand(rest*_linearise(data.trig[0], data.trig[1])), n, z)
    if data.impulse is not None:
        value = (term/KroneckerDelta(n, Integer(data.impulse))).subs(n, Integer(data.impulse))
        return value*pow(z, Integer(-data.impulse))
    a = data.base
    special = int(data.factorial) + int(data.shift is not None) + len(data.trig)
    if special > 1:
        raise NotImplementedError('the Z-transform of %s' % (term,))
    rational = True
    if data.factorial:
        image: Expr = exp(a/z)
        rational = False
    elif data.shift is not None:
        m = data.shift
        ratio = a/z
        image = pow(z/a, Integer(m))*(-log(1 - ratio) - add(*[pow(ratio, Integer(j))/j for j in range(1, m)]))
        rational = False
    elif data.trig:
        function = data.trig[0]
        parts = _linear(function.args[0], n) if isinstance(function.args[0], Expr) else None
        if parts is None:
            raise NotImplementedError('the Z-transform of %s' % (term,))
        omega, phase = parts
        denominator = z**2 - 2*a*z*cos(omega) + a**2
        even, odd = z*(z - a*cos(omega)), a*z*sin(omega)
        if isinstance(function, Cos):
            image = (cos(phase)*even - sin(phase)*odd)/denominator
        else:
            image = (sin(phase)*even + cos(phase)*odd)/denominator
    else:
        image = z/(z - a)
    for _ in range(data.power):
        image = -z*image.diff(z)
    result = data.coefficient*image
    return normal(result) if rational else result


def z_transform(f: ExprLike, n: Symbol, z: Symbol) -> Expr:
    r"""The unilateral Z-transform $\sum_{n \ge 0} f(n) z^{-n}$ of the
    sequence ``f`` in ``n``, for ``|z|`` large enough;
    ``NotImplementedError`` when a term is in none of the known families
    (see the module documentation).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import sin, KroneckerDelta
    >>> from sympy_extras.concrete.ztransform import z_transform
    >>> n, z, w = Symbol('n'), Symbol('z'), Symbol('w')
    >>> z_transform(1, n, z), z_transform(n**2, n, z), z_transform(3*KroneckerDelta(n, 2) + 1/(n + 1), n, z)
    (z/(z - 1), (z**2 + z)/(z**3 - 3*z**2 + 3*z - 1), -z*log(1 - 1/z) + 3/z**2)
    >>> z_transform(sin(w*n), n, z)
    z*sin(w)/(z**2 - 2*z*cos(w) + 1)
    >>> from sympy_extras.functions import Heaviside
    >>> z_transform(2**n*Heaviside(n - 3, 1), n, z)          # 2**n from n = 3 on
    8/(z**3 - 2*z**2)
    """
    expression = expand(to_expr(f))
    terms = expression.args if isinstance(expression, Add) else (expression,)
    rational: list[Expr] = []
    others: list[Expr] = []
    for term in terms:
        if not isinstance(term, Expr):
            raise TypeError('%s is not an expression' % (term,))
        image = _transform_term(term, n, z)
        if image.has(Exp) and image.has(z) and any(isinstance(node, Exp) and node.has(z) for node in _nodes(image)):
            others.append(image)
        elif any(isinstance(node, type(log(z))) and node.has(z) for node in _nodes(image)):
            others.append(image)
        else:
            rational.append(image)
    total = normal(add(*rational)) if rational else _ZERO
    return add(total, *others)


def _nodes(e: Expr) -> list[Expr]:
    found: list[Expr] = [e]
    for a in e.args:
        if isinstance(a, Expr):
            found.extend(_nodes(a))
    return found


# ---------------------------------------------------------------------
# inverse

class _Part:
    """The contribution of the poles which are the roots of the
    irreducible polynomial ``factor``: ``sum over its roots t of sum_j
    coefficients[j-1](t) * binomial(n + j - 1, j - 1) * t**n``, with
    ``factor`` and the coefficients polynomials in ``symbol``."""

    def __init__(self, factor: Expr, symbol: Symbol, coefficients: Sequence[Expr]) -> None:
        self.factor: Final[Expr] = factor
        self.symbol: Final[Symbol] = symbol
        self.coefficients: Final[tuple[Expr, ...]] = tuple(coefficients)


def _field_polynomial(field: _Field[T], coefficients: Sequence[Expr]) -> list[T]:
    return densebasic.dup_strip([field.element(c) for c in coefficients], field.domain)


def _polynomial_expr(field: _Field[T], f: Sequence[T], x: Expr) -> Expr:
    degree = len(f) - 1
    return add(*[mul(field.expr(c), pow(x, Integer(degree - k))) for k, c in enumerate(f)])


def _decompose(field: _Field[T], numer: Sequence[Expr], denom: Sequence[Expr], t: Symbol, parameters: Sequence[Expr]
               ) -> tuple[list[Expr], list[_Part]]:
    """The impulses (the coefficient of ``x**j`` of the polynomial part)
    and the parts of ``A(x)/B(x)``, ``x = 1/z``; ``numer`` and ``denom``
    are the coefficients in ``z``, lowest degree first."""
    K = field.domain
    width = max(len(numer), len(denom))
    A = _field_polynomial(field, list(numer) + [_ZERO]*(width - len(numer)))
    B = _field_polynomial(field, list(denom) + [_ZERO]*(width - len(denom)))
    if len(denom) < width or not B or K.is_zero(B[-1]):
        raise ValueError('not the Z-transform of a sequence: the function is not finite at infinity')
    quotient, remainder = densearith.dup_div(A, B, K)
    impulses = [field.expr(c) for c in reversed(quotient)]
    if not remainder or len(B) == 1:
        return impulses, []
    leading = K.invert(B[0])
    remainder = densearith.dup_mul_ground(remainder, leading, K)
    expression = _polynomial_expr(field, densearith.dup_mul_ground(B, leading, K), t)
    numerator = together(expression).as_numer_denom()[0] if parameters else expression
    _, listed = factor_list(numerator, t, *parameters)
    factors: list[tuple[list[T], Expr, int]] = []
    for factor, exponent in listed:
        if not factor.has(t) or not isinstance(exponent, Integer):
            continue
        g = _field_polynomial(field, Poly(factor, t).all_coeffs())
        factors.append((densearith.dup_mul_ground(g, K.invert(g[0]), K), factor, exponent.p))
    powers = [densearith.dup_pow(g, m, K) for g, _, m in factors]
    parts: list[_Part] = []
    for index, (g, factor, m) in enumerate(factors):
        cofactor = [K.one]
        for other, power in enumerate(powers):
            if other != index:
                cofactor = densearith.dup_mul(cofactor, power, K)
        inverse = euclidtools.dup_invert(densearith.dup_rem(cofactor, powers[index], K), powers[index], K) if len(factors) > 1 else [K.one]
        local = densearith.dup_rem(densearith.dup_mul(remainder, inverse, K), powers[index], K)
        # the expansion of local/g**m at a root of g, in F = K[t]/(g*), with
        # g* the reciprocal polynomial: t is the pole 1/root in z
        integral = Poly(factor, t).all_coeffs()
        reciprocal_expr = add(*[mul(c, pow(t, Integer(k))) for k, c in enumerate(integral)])
        reciprocal = _field_polynomial(field, list(reversed(integral)))
        F = QuotientField(K, densearith.dup_mul_ground(reciprocal, K.invert(reciprocal[0]), K))
        pole = F.generator
        root = F.invert(pole)
        lifted = [F.from_ground(c) for c in g]
        cofactor_at_root = densearith.dup_div(lifted, [F.one, F.neg(root)], F)[0]
        shift = [F.one, root]

        def shifted(f: Sequence[list[T]]) -> list[list[T]]:
            total: list[list[T]] = []
            for c in f:
                total = densearith.dup_add(densearith.dup_mul(total, shift, F), densebasic.dup_strip([c], F), F)
            return total

        top = list(reversed(shifted([F.from_ground(c) for c in local])))
        bottom = list(reversed(densearith.dup_pow(shifted(cofactor_at_root), m, F)))
        series: list[list[T]] = []
        scale = F.invert(bottom[0])
        for k in range(m):
            value = top[k] if k < len(top) else F.zero
            for i in range(1, k + 1):
                if i < len(bottom):
                    value = F.sub(value, F.mul(bottom[i], series[k - i]))
            series.append(F.mul(value, scale))
        minus_inverse = F.neg(pole)
        coefficients: list[Expr] = []
        for j in range(1, m + 1):
            element = F.mul(series[m - j], F.pow(minus_inverse, j))
            coefficients.append(_polynomial_expr(field, element, t))
        parts.append(_Part(reciprocal_expr, t, coefficients))
    return impulses, parts


def _binomial(n: Expr, j: int) -> Expr:
    """``binomial(n + j - 1, j - 1)`` as a polynomial in ``n``."""
    if j == 1:
        return _ONE
    denominator = 1
    for i in range(1, j):
        denominator *= i
    return mul(*[n + i for i in range(1, j)])/denominator


def _real_form(part: _Part, n: Expr) -> Optional[Expr]:
    """A quadratic factor with rational coefficients and complex roots,
    with cosines and sines."""
    polynomial = Poly(part.factor, part.symbol)
    if polynomial.degree() != 2:
        return None
    values = polynomial.all_coeffs()
    if not all(isinstance(c, Rational) for c in values):
        return None
    lead, p, q = (c.as_fraction() for c in values if isinstance(c, Rational))
    p, q = p/lead, q/lead
    discriminant = q - p*p/4
    if discriminant <= 0:
        return None
    # the pole t = -p/2 + i*s = sqrt(q) exp(i theta): t**n = q**(n/2) (cos(n theta) + i sin(n theta))
    s = sqrt(number_from_fraction(discriminant))
    modulus = sqrt(number_from_fraction(q))
    theta = acos(number_from_fraction(-p/2)/modulus)
    total: list[Expr] = []
    for j, coefficient in enumerate(part.coefficients, 1):
        linear = Poly(coefficient, part.symbol).all_coeffs() if coefficient.has(part.symbol) else [coefficient]
        e1, e0 = (linear[0], linear[1]) if len(linear) == 2 else (_ZERO, linear[0])
        u = e0 - e1*number_from_fraction(p/2)
        v = e1*s
        total.append(2*_binomial(n, j)*(u*cos(n*theta) - v*sin(n*theta)))
    return pow(modulus, n)*add(*total)


def _assemble(impulses: Sequence[Expr], parts: Sequence[_Part], n: Expr, real: bool) -> Expr:
    terms: list[Expr] = [c*KroneckerDelta(n, Integer(j)) for j, c in enumerate(impulses) if c != 0]
    for part in parts:
        if real:
            form = _real_form(part, n)
            if form is not None:
                terms.append(form)
                continue
        degree = Poly(part.factor, part.symbol).degree()
        found = roots(part.factor, part.symbol)
        if not isinstance(found, dict) or sum(found.values()) != degree:
            raise NotImplementedError('the roots of %s are not expressed in radicals' % (part.factor,))
        for value in sorted(found, key=default_sort_key):
            for j, coefficient in enumerate(part.coefficients, 1):
                terms.append(coefficient.subs(part.symbol, value)*_binomial(n, j)*pow(value, n))
    return add(*terms)


def inverse_z_transform(F: ExprLike, z: Symbol, n: Symbol) -> Expr:
    r"""The sequence $f(n)$, $n \ge 0$, whose Z-transform is the rational
    function ``F`` of ``z`` (with symbols as parameters): the coefficient
    of $z^{-n}$ of its expansion at infinity.

    ``ValueError`` when ``F`` is not finite at infinity, and
    ``NotImplementedError`` when it is not a rational function of ``z``
    and symbols or when the poles are not found in radicals.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.concrete.ztransform import inverse_z_transform
    >>> n, z = Symbol('n'), Symbol('z')
    >>> inverse_z_transform(z/((z - 1)*(z - 2)), z, n), inverse_z_transform(z*(z + 1)/(z - 1)**3, z, n)
    (2**n - 1, n**2)
    >>> inverse_z_transform(z/(z**2 - z + 1), z, n)
    2*sqrt(3)*sin(pi*n/3)/3
    """
    expression = together(to_expr(F))
    numerator, denominator = expression.as_numer_denom()
    kernels = _kernels([ExprMatrix(1, 2, [numerator, denominator])])
    if not all(isinstance(k, Symbol) for k in kernels):
        raise NotImplementedError('the inverse Z-transform of %s, which is not a rational function of symbols' % (expression,))
    used = {k.name for k in kernels if isinstance(k, Symbol)}
    name, count = '_t', 0
    while name in used:
        name, count = '_t%d' % count, count + 1
    t = Symbol(name)
    numer = list(reversed(Poly(numerator, z).all_coeffs())) if numerator.has(z) else [numerator]
    denom = list(reversed(Poly(denominator, z).all_coeffs())) if denominator.has(z) else [denominator]
    parameters = [k for k in kernels if k != z]
    if parameters:
        impulses, parts = _decompose(_FunctionField(_PolynomialLift(parameters)), numer, denom, t, parameters)
    else:
        impulses, parts = _decompose(_RationalField(), numer, denom, t, parameters)
    result = _assemble(impulses, parts, n, not parameters)
    return expand(result) if not parameters else result
