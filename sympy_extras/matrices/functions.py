r"""Functions of matrices: the exponential, symbolic powers, and any
function given by an expression.

For a matrix $A$ with characteristic polynomial $\chi = \prod_\lambda (x
- \lambda)^{m_\lambda}$ and a function $f$ with enough derivatives at the
eigenvalues,

.. math:: f(A) = \sum_{\lambda} \sum_{j=0}^{m_\lambda - 1}
          f^{(j)}(\lambda)\, C_{\lambda, j}, \qquad
          C_{\lambda, j} = \frac{(A - \lambda)^j}{j!}\, E_\lambda,

where $E_\lambda$ is the projector on the generalised eigenspace of
$\lambda$ along the others; the $C_{\lambda, j}$ are the *component
matrices* of $A$, which do not depend on $f$ [1]_, [2]_. The projector is
a polynomial in $A$: with $\chi = (x - \lambda)^m c(x)$ and $v\, c \equiv
1 \pmod{(x - \lambda)^m}$, $E_\lambda = (v c)(A)$.

As for the eigenvectors (:mod:`sympy_extras.matrices.eigen`), the
components are computed once for all the roots of an irreducible factor
$g$ of $\chi$, in the field $K[t]/(g)$ where $t$ is such a root: their
entries are polynomials in $t$ over the field $K$ of the entries. No
matrix of eigenvectors is inverted and no expression in radicals is
tested for zero; the eigenvalues are substituted at the end.

:func:`matrix_exp` writes the pair of complex eigenvalues of a quadratic
factor with rational coefficients with cosines and sines.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.functions import matrix_exp, matrix_power, matrix_function
>>> t, n, x = Symbol('t'), Symbol('n'), Symbol('x')
>>> matrix_exp(ExprMatrix.from_rows([[0, 1], [-1, 0]]), t)
ExprMatrix([[cos(t), sin(t)], [-sin(t), cos(t)]])
>>> matrix_exp(ExprMatrix.from_rows([[1, 1], [0, 1]]), t)
ExprMatrix([[exp(t), t*exp(t)], [0, exp(t)]])
>>> matrix_power(ExprMatrix.from_rows([[2, 1], [0, 2]]), n)
ExprMatrix([[2**n, 2**n*n/2], [0, 2**n]])
>>> matrix_function(ExprMatrix.from_rows([[4, 0], [0, 9]]), x**2 + 1/x, x)
ExprMatrix([[65/4, 0], [0, 730/9]])

References
==========

.. [1] F. R. Gantmacher, *The Theory of Matrices*, vol. 1, Chelsea, 1959,
       Chapter V (functions of matrices, the components of a matrix).
.. [2] N. J. Higham, *Functions of Matrices: Theory and Computation*,
       SIAM, 2008, Sections 1.2 and 1.2.2 (the Jordan and the polynomial
       interpolation definitions).
"""
from __future__ import annotations

from typing import Final, Optional, Sequence, TypeVar

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.numbers import Integer, Rational, number_from_fraction
from sympy_extras.core.power import pow, sqrt
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions import cos, exp, sin
from sympy_extras.polys import densearith, euclidtools
from sympy_extras.polys.domains.quotientfield import QuotientField

from .dense import ExprMatrix
from .domainmatrix import DomainMatrix
from .eigen import (_Field, _FunctionField, _GaussianField, _RationalField, _characteristic_factors, _coefficients, _column,
                    _extension, _fresh_symbol, _roots_in_radicals, _shifted)
from .linalg import _PolynomialLift, _complex_lift

__all__ = ['MatrixComponents', 'matrix_components', 'matrix_function', 'matrix_exp', 'matrix_power']

T = TypeVar('T')


class MatrixComponents:
    """The component matrices of the eigenvalues which are the roots of
    the irreducible ``factor`` (in ``symbol``): ``components[j]`` is
    ``C_j`` with ``symbol`` standing for the eigenvalue, and ``f(M)`` is
    the sum over the eigenvalues of ``f^(j)(eigenvalue) * C_j``."""

    __slots__ = ('factor', 'symbol', 'components')

    def __init__(self, factor: Expr, symbol: Symbol, components: Sequence[ExprMatrix]) -> None:
        self.factor: Final[Expr] = factor
        self.symbol: Final[Symbol] = symbol
        self.components: Final[tuple[ExprMatrix, ...]] = tuple(components)

    def values(self) -> Optional[list[Expr]]:
        """The roots of ``factor`` in radicals, or ``None``."""
        return _roots_in_radicals(self.factor, self.symbol)

    def __repr__(self) -> str:
        return 'MatrixComponents(%s, %r)' % (self.factor, list(self.components))


def _components(field: _Field[T], M: ExprMatrix, factors: Sequence[tuple[Expr, int]], unknown: Symbol, x: Symbol
                ) -> list[MatrixComponents]:
    K = field.domain
    n = M.rows
    # the characteristic polynomial, monic, over K
    monic: list[list[T]] = []
    characteristic = [K.one]
    for factor, multiplicity in factors:
        coefficients = [field.element(c) for c in _coefficients(factor, unknown)]
        leading = K.invert(coefficients[0])
        g = [K.mul(c, leading) for c in coefficients]
        monic.append(g)
        characteristic = densearith.dup_mul(characteristic, densearith.dup_pow(g, multiplicity, K), K)
    result: list[MatrixComponents] = []
    for (factor, multiplicity), g in zip(factors, monic):
        F: QuotientField[T] = _extension(field, factor, unknown)
        lifted = [F.from_ground(c) for c in characteristic]
        power = densearith.dup_pow([F.one, F.neg(F.generator)], multiplicity, F)
        cofactor = densearith.dup_div(lifted, power, F)[0]
        if len(cofactor) == 1:
            projector_polynomial = [F.one]
        else:
            inverse = euclidtools.dup_invert(densearith.dup_rem(cofactor, power, F), power, F)
            projector_polynomial = densearith.dup_rem(densearith.dup_mul(inverse, cofactor, F), lifted, F)
        A = DomainMatrix([[F.from_ground(field.element(e)) for e in M.row(i)] for i in range(n)], F, n)
        identity = DomainMatrix.eye(n, F)
        projector = DomainMatrix.zeros(n, n, F)
        for c in projector_polynomial:
            projector = projector * A + identity.scale(c)
        shifted = _shifted(field, F, M)
        components: list[ExprMatrix] = []
        current = projector
        for j in range(multiplicity):
            if j:
                current = (shifted * current).scale(F.invert(F.from_int(j)))
            columns = [_column(field, [current.rows[i][k] for i in range(n)], x) for k in range(n)]
            components.append(ExprMatrix(n, n, [columns[k][i, 0] for i in range(n) for k in range(n)]))
        result.append(MatrixComponents(factor.subs(unknown, x), x, components))
    return result


def matrix_components(M: ExprMatrix, x: Symbol) -> list[MatrixComponents]:
    """The component matrices, grouped by the irreducible factors of the
    characteristic polynomial (in ``x``).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.functions import matrix_components
    >>> x = Symbol('x')
    >>> matrix_components(ExprMatrix.from_rows([[1, 1], [1, 0]]), x)
    [MatrixComponents(x**2 - x - 1, [ExprMatrix([[x/5 + 2/5, 2*x/5 - 1/5], [2*x/5 - 1/5, 3/5 - x/5]])])]
    """
    unknown = _fresh_symbol(M)
    kernels, factors = _characteristic_factors(M, unknown)
    gaussian = _complex_lift(kernels)
    if gaussian is not None:
        return _components(_GaussianField(gaussian), M, factors, unknown, x)
    if kernels:
        return _components(_FunctionField(_PolynomialLift(kernels)), M, factors, unknown, x)
    return _components(_RationalField(), M, factors, unknown, x)


def _explicit_values(group: MatrixComponents) -> list[Expr]:
    values = group.values()
    if values is None:
        raise NotImplementedError('the roots of %s are not expressed in radicals: see matrix_components' % (group.factor,))
    return values


def matrix_function(M: ExprMatrix, f: ExprLike, x: Symbol) -> ExprMatrix:
    """``f(M)`` for the function ``f`` given as an expression in ``x``:
    the sum over the eigenvalues of the derivatives of ``f`` there times
    the component matrices. ``NotImplementedError`` when the eigenvalues
    are not expressed in radicals.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.functions import matrix_function
    >>> x = Symbol('x')
    >>> matrix_function(ExprMatrix.from_rows([[5, 4], [4, 5]]), sqrt(x), x)     # a square root of the matrix
    ExprMatrix([[2, 1], [1, 2]])
    >>> matrix_function(ExprMatrix.from_rows([[1, 1], [0, 1]]), 1/x, x)
    ExprMatrix([[1, -1], [0, 1]])
    """
    function = to_expr(f)
    n = M.rows
    symbol = _fresh_symbol(M)
    used = {s.name for s in function.free_symbols if isinstance(s, Symbol)}
    while symbol.name in used:
        symbol = Symbol(symbol.name + '_')
    zero: Expr = Integer(0)
    total = [zero] * (n*n)
    for group in matrix_components(M, symbol):
        values = _explicit_values(group)
        derivative = function
        for j, component in enumerate(group.components):
            if j:
                derivative = derivative.diff(x)
            for value in values:
                scalar = derivative.subs(x, value)
                matrix = component.subs(symbol, value)
                total = [a + scalar*b for a, b in zip(total, matrix.entries)]
    return ExprMatrix(n, n, [e.expand() for e in total])


def _real_exponential(group: MatrixComponents, t: Expr) -> Optional[list[Expr]]:
    """The contribution to ``exp(t*M)`` of a quadratic factor with
    rational coefficients and complex roots ``alpha +- i*beta``:
    ``2 exp(alpha t) sum_j t**j (Re C_j cos(beta t) - Im C_j sin(beta
    t))``."""
    coefficients = _coefficients(group.factor, group.symbol)
    if len(coefficients) != 3 or not all(isinstance(c, Rational) for c in coefficients):
        return None
    lead, p, q = (c.as_fraction() for c in coefficients if isinstance(c, Rational))
    p, q = p/lead, q/lead
    discriminant = q - p*p/4
    if discriminant <= 0:
        return None
    alpha = number_from_fraction(-p/2)
    beta = sqrt(number_from_fraction(discriminant))
    size = len(group.components[0].entries)
    zero: Expr = Integer(0)
    total = [zero] * size
    for j, component in enumerate(group.components):
        for index, entry in enumerate(component.entries):
            linear = _coefficients(entry, group.symbol) if entry.has(group.symbol) else [entry]
            if len(linear) > 2:
                raise ArithmeticError('a component which is not reduced: %s' % (entry,))
            c1, c0 = (linear[0], linear[1]) if len(linear) == 2 else (zero, linear[0])
            real, imaginary = c0 + c1*alpha, c1*beta
            total[index] = total[index] + 2*pow(t, Integer(j))*(real*cos(beta*t) - imaginary*sin(beta*t))
    growth = exp(alpha*t)
    return [growth*e for e in total]


def matrix_exp(M: ExprMatrix, t: ExprLike = 1) -> ExprMatrix:
    r"""The matrix exponential $e^{tM}$, the fundamental matrix of the
    system $y' = M y$. ``NotImplementedError`` when the eigenvalues are
    not expressed in radicals.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.functions import matrix_exp
    >>> t, a = Symbol('t'), Symbol('a')
    >>> matrix_exp(ExprMatrix.from_rows([[0, 1], [0, 0]]), t), matrix_exp(ExprMatrix.from_rows([[a, 1], [0, a]]))
    (ExprMatrix([[1, t], [0, 1]]), ExprMatrix([[exp(a), exp(a)], [0, exp(a)]]))
    >>> matrix_exp(ExprMatrix.from_rows([[1, -2], [2, 1]]), t)
    ExprMatrix([[exp(t)*cos(2*t), -exp(t)*sin(2*t)], [exp(t)*sin(2*t), exp(t)*cos(2*t)]])
    """
    time = to_expr(t)
    n = M.rows
    symbol = _fresh_symbol(M)
    used = {s.name for s in time.free_symbols if isinstance(s, Symbol)}
    while symbol.name in used:
        symbol = Symbol(symbol.name + '_')
    zero: Expr = Integer(0)
    total = [zero] * (n*n)
    numbers = all(isinstance(e, Rational) for e in M.entries)
    for group in matrix_components(M, symbol):
        real = _real_exponential(group, time) if numbers else None
        if real is not None:
            total = [a + b for a, b in zip(total, real)]
            continue
        for value in _explicit_values(group):
            growth = exp(time*value)
            for j, component in enumerate(group.components):
                scalar = pow(time, Integer(j))*growth
                matrix = component.subs(symbol, value)
                total = [a + scalar*b for a, b in zip(total, matrix.entries)]
    return ExprMatrix(n, n, [add(e.expand()) for e in total])


def matrix_power(M: ExprMatrix, n: ExprLike) -> ExprMatrix:
    """``M**n`` for a symbolic exponent ``n``, as a closed form in ``n``
    (valid for the integers ``n >= 0`` when ``M`` is invertible; with the
    eigenvalue zero, for ``n`` at least its multiplicity, the powers
    ``0**n`` being left as they are).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.functions import matrix_power
    >>> n = Symbol('n')
    >>> matrix_power(ExprMatrix.from_rows([[1, 1], [0, 1]]), n), matrix_power(ExprMatrix.from_rows([[3, 0], [0, 2]]), n)
    (ExprMatrix([[1, n], [0, 1]]), ExprMatrix([[3**n, 0], [0, 2**n]]))
    """
    exponent = to_expr(n)
    variable = Symbol('_base')
    used = {s.name for s in exponent.free_symbols if isinstance(s, Symbol)} | {
        s.name for e in M.entries for s in e.free_symbols if isinstance(s, Symbol)}
    while variable.name in used:
        variable = Symbol(variable.name + '_')
    return matrix_function(M, pow(variable, exponent), variable)
