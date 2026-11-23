r"""Eigenvalues and eigenvectors of matrices of expressions.

The characteristic polynomial (Berkowitz' algorithm,
:mod:`sympy_extras.matrices.linalg`) is factored into irreducible factors
over the field $K$ of rational functions in the symbols of the matrix.
For a factor $f$ the eigenvectors are computed *once for all its roots*:
the null space of $M - \lambda I$ is found by elimination over the field
$K[\lambda]/(f)$ (:mod:`sympy_extras.polys.domains.quotientfield`), where
$\lambda$ is a root of $f$ by construction, so that the elimination
never has to decide whether an expression in radicals is zero. The
vectors are polynomials in $\lambda$; substituting a root of $f$ gives
the eigenvectors of that eigenvalue.

This makes the structure independent of solving $f$: the algebraic and
geometric multiplicities, and so whether the matrix is diagonalisable,
are known for every matrix (:func:`eigenspaces`,
:func:`is_diagonalizable`). The eigenvalues themselves are written out
(:func:`eigenvalues`, :func:`eigenvectors`, :func:`diagonalize`) when the
roots of the factors are found in radicals
(:func:`sympy_extras.polys.polyroots.roots`).

The Jordan form is found in the same way (:func:`jordan_chains`,
:func:`jordan_form`): over $K[\lambda]/(f)$ the kernels of the powers of
$M - \lambda I$ give the sizes of the blocks, and the chains of
generalised eigenvectors are built from the longest down, the vectors of
each level completing the kernel of the previous power together with the
images of the level above [3]_.

The entries are rational functions of symbols, possibly with the
imaginary unit and square roots of rational numbers, which $K$ then
contains. The characteristic polynomial is factored over $\mathbb{Q}$,
or over $\mathbb{Q}(i)$ when ``I`` remains in it (Trager's algorithm,
:mod:`sympy_extras.polys.gaussianfactor`); when square roots remain in
its coefficients, or with other kernels, the factorisation over the
field they generate would be needed (``NotImplementedError``).

**Floating point entries** are the rational numbers they store (see
:mod:`sympy_extras.matrices.linalg`): the characteristic polynomial, its
factorisation and the eliminations are exact, so that the multiplicities,
the dimensions of the eigenspaces and the Jordan structure are those of
the matrix as stored, and no eigenvector is "independent" by a rounding
error (``[[1.0, 1.0], [0, 1.0]]`` has the eigenvalue ``1.0`` twice, one
eigenvector, and is not diagonalisable). The eigenvalues are then the
roots of the irreducible factors, isolated exactly
(:func:`sympy_extras.polys.complexrootof.all_roots`: any degree) and
written as floats of the precision of the entries, in increasing order
of their real parts; :func:`eigenspaces` and :func:`jordan_chains` return
the exact structure.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.eigen import eigenvalues, eigenvectors, eigenspaces, is_diagonalizable, diagonalize
>>> a, x = Symbol('a'), Symbol('x')
>>> M = ExprMatrix.from_rows([[2, 1], [1, 2]])
>>> eigenvalues(M)
{1: 1, 3: 1}
>>> eigenvectors(M)
[(1, 1, [ExprMatrix([[-1], [1]])]), (3, 1, [ExprMatrix([[1], [1]])])]
>>> diagonalize(M)
(ExprMatrix([[-1, 1], [1, 1]]), ExprMatrix([[1, 0], [0, 3]]))
>>> F = ExprMatrix.from_rows([[1, 1], [1, 0]])
>>> eigenspaces(F, x)
[Eigenspace(x**2 - x - 1, 1, [ExprMatrix([[x], [1]])])]
>>> eigenvalues(F)
{1/2 - sqrt(5)/2: 1, 1/2 + sqrt(5)/2: 1}
>>> J = ExprMatrix.from_rows([[a, 1], [0, a]])
>>> eigenspaces(J, x), is_diagonalizable(J)
([Eigenspace(-a + x, 2, [ExprMatrix([[1], [0]])])], False)

References
==========

.. [1] K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
       Algebra*, Kluwer, 1992, Sections 8.8 and 9.
.. [2] S. J. Berkowitz, *On computing the determinant in small parallel
       time using a small number of processors*, Information Processing
       Letters 18 (1984), 147-150.
.. [3] R. A. Horn, C. R. Johnson, *Matrix Analysis*, 2nd ed., Cambridge
       University Press, 2013, Section 3.1 (the Jordan canonical form:
       the kernels of the powers of the nilpotent part and the chains).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Final, Generic, Optional, Sequence, TypeVar

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import mul
from sympy_extras.core.function import expand
from sympy_extras.core.numbers import Float, ImaginaryUnit, Integer, Rational, number_from_fraction
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.domains.quotientfield import QuotientField
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.complexrootof import all_roots
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.domains.gaussiandomains import QQ_I
from sympy_extras.polys.gaussianfactor import gaussian_ext_factor
from sympy_extras.polys.polyerrors import PolynomialError
from sympy_extras.polys.polytools import factor_list
from sympy_extras.polys.rings import PolyRing

from .dense import ExprMatrix
from .domainmatrix import DomainMatrix
from .linalg import (_ComplexLift, _IntegerLift, _PolynomialLift, _Radical, _complex_lift, _is_radical, _kernels, _square,
                     approximate, approximate_matrix, charpoly, exact_matrix, float_precision)

__all__ = ['Eigenspace', 'eigenspaces', 'eigenvalues', 'eigenvectors', 'is_diagonalizable', 'diagonalize', 'JordanChains',
           'jordan_chains', 'jordan_form']

T = TypeVar('T')


def _coefficients(e: Expr, x: Symbol) -> list[Expr]:
    """The coefficients of a polynomial in ``x``, highest degree first
    (the coefficients may contain ``I``)."""
    expanded = expand(e)
    found: dict[int, list[Expr]] = {}
    for term in (expanded.args if isinstance(expanded, Add) else (expanded,)):
        if not isinstance(term, Expr):
            raise TypeError('%s is not an expression' % (term,))
        coefficient, power = term.as_independent(x)
        if power == 1:
            degree = 0
        elif power == x:
            degree = 1
        elif isinstance(power, Pow) and power.base == x and isinstance(power.exp, Integer) and power.exp.p > 0:
            degree = power.exp.p
        else:
            raise ValueError('%s is not a polynomial in %s' % (e, x))
        found.setdefault(degree, []).append(coefficient)
    top = max(found) if found else 0
    return [add(*found.get(k, [])) for k in range(top, -1, -1)]


def _roots_in_radicals(factor: Expr, x: Symbol) -> Optional[list[Expr]]:
    """The roots of an irreducible factor in radicals, or ``None``."""
    coefficients = _coefficients(factor, x)
    if len(coefficients) == 2:
        return [-coefficients[1]/coefficients[0]]
    if any(c.has(ImaginaryUnit) for c in coefficients):
        if len(coefficients) != 3:
            return None
        a, b, c = coefficients
        root = sqrt(expand(b*b - 4*a*c))
        return sorted([(-b - root)/(2*a), (-b + root)/(2*a)], key=default_sort_key)
    try:
        found = roots(factor, x)
    except PolynomialError:
        return None
    if not isinstance(found, dict) or sum(found.values()) != len(coefficients) - 1:
        return None
    return sorted(found, key=default_sort_key)


def _isolated_roots(factor: Expr, x: Symbol) -> Optional[list[Expr]]:
    """The roots of an irreducible factor as exact numbers for a
    numerical result: isolated roots (nodes of any degree) when the
    coefficients are rational numbers, the radical forms otherwise."""
    if factor.free_symbols == {x} and not factor.has(ImaginaryUnit):
        return all_roots(factor, x, radicals=False)
    found = _roots_in_radicals(factor, x)
    if found is None and factor.free_symbols == {x}:
        return _gaussian_roots(factor, x)
    return found


_Gaussian = tuple[Fraction, Fraction]


def _gaussian_value(coefficients: Sequence[_Gaussian], point: _Gaussian) -> Fraction:
    """The square of the absolute value of the polynomial at the point
    (Horner's rule on pairs of rational numbers)."""
    re, im = Fraction(0), Fraction(0)
    for a, b in coefficients:
        re, im = re*point[0] - im*point[1] + a, re*point[1] + im*point[0] + b
    return re*re + im*im


def _gaussian_roots(factor: Expr, x: Symbol) -> Optional[list[Expr]]:
    """The roots of a factor with coefficients in ``QQ(I)`` which is
    irreducible there and not real, of any degree, as isolated roots of
    its norm ``factor*conjugate(factor)``, a polynomial with rational
    coefficients; ``None`` when they are not told apart.

    The factor and its conjugate have no common root, so every root of
    the norm is a root of exactly one of them: the one whose value at a
    rational approximation of the root is smaller by more than half of
    the digits of the approximation (the other value is not zero).
    """
    pairs: list[_Gaussian] = []
    for c in _coefficients(expand(factor), x):
        re, im = c.as_real_imag()
        if not isinstance(re, Rational) or not isinstance(im, Rational):
            return None
        pairs.append((Fraction(re.p, re.q), Fraction(im.p, im.q)))
    conjugate = [(a, -b) for a, b in pairs]
    degree = len(pairs) - 1
    I = ImaginaryUnit()
    conjugated = add(*[mul(number_from_fraction(a) + number_from_fraction(b)*I, pow(x, Integer(degree - k)))
                       for k, (a, b) in enumerate(conjugate)])
    norm = expand(factor*conjugated)
    if norm.has(ImaginaryUnit):
        return None
    candidates: list[Expr] = []
    for root in all_roots(norm, x, radicals=False):
        if root not in candidates:
            candidates.append(root)
    chosen: list[Expr] = []
    for root in candidates:
        verdict: Optional[bool] = None
        for digits in (30, 60, 120, 240):
            re, im = root.evalf(digits).as_real_imag()
            if not isinstance(re, (Float, Rational)) or not isinstance(im, (Float, Rational)):
                return None
            point = (re.as_fraction() if isinstance(re, Float) else Fraction(re.p, re.q),
                     im.as_fraction() if isinstance(im, Float) else Fraction(im.p, im.q))
            mine, other = _gaussian_value(pairs, point), _gaussian_value(conjugate, point)
            scale = Fraction(10)**digits        # the squares: half of the digits apart
            if mine*scale < other:
                verdict = True
                break
            if other*scale < mine:
                verdict = False
                break
        if verdict is None:
            return None
        if verdict:
            chosen.append(root)
    return chosen if len(chosen) == degree else None


def _numerical_order(value: Expr) -> tuple[float, float]:
    number = complex(value)
    return (number.real, number.imag)


class Eigenspace:
    """The eigenvalues which are the roots of ``factor``, an irreducible
    factor of the characteristic polynomial in ``symbol``, of multiplicity
    ``multiplicity`` in it; ``basis`` is a basis of the eigenspace of each
    of them, with ``symbol`` standing for the eigenvalue."""

    __slots__ = ('factor', 'symbol', 'multiplicity', 'basis')

    def __init__(self, factor: Expr, symbol: Symbol, multiplicity: int, basis: Sequence[ExprMatrix]) -> None:
        self.factor: Final[Expr] = factor
        self.symbol: Final[Symbol] = symbol
        self.multiplicity: Final[int] = multiplicity
        self.basis: Final[tuple[ExprMatrix, ...]] = tuple(basis)

    @property
    def geometric_multiplicity(self) -> int:
        return len(self.basis)

    def values(self) -> Optional[list[Expr]]:
        """The roots of ``factor`` in radicals, or ``None`` when they are
        not found."""
        return _roots_in_radicals(self.factor, self.symbol)

    def vectors(self, value: Expr) -> list[ExprMatrix]:
        """The basis of the eigenspace of the eigenvalue ``value``."""
        return [v.subs(self.symbol, value) for v in self.basis]

    def isolated_values(self) -> Optional[list[Expr]]:
        """The roots of ``factor`` as exact numbers which have a
        numerical value at any precision: isolated roots
        (:class:`~sympy_extras.polys.complexrootof.ComplexRootOf`) when
        its coefficients are rational numbers, the radical forms of
        :meth:`values` otherwise, and for a factor with ``I`` in its
        coefficients which has none (a degree above 2) the isolated roots
        of its norm which are its own."""
        return _isolated_roots(self.factor, self.symbol)

    def __repr__(self) -> str:
        return 'Eigenspace(%s, %d, %r)' % (self.factor, self.multiplicity, list(self.basis))


class JordanChains:
    """The Jordan blocks of the eigenvalues which are the roots of the
    irreducible ``factor`` (in ``symbol``): each chain ``v_1, ..., v_k``
    has ``(M - symbol) v_1 = 0`` and ``(M - symbol) v_i = v_(i-1)``, and
    gives a block of size ``k`` for every root."""

    __slots__ = ('factor', 'symbol', 'multiplicity', 'chains')

    def __init__(self, factor: Expr, symbol: Symbol, multiplicity: int, chains: Sequence[Sequence[ExprMatrix]]) -> None:
        self.factor: Final[Expr] = factor
        self.symbol: Final[Symbol] = symbol
        self.multiplicity: Final[int] = multiplicity
        self.chains: Final[tuple[tuple[ExprMatrix, ...], ...]] = tuple(tuple(chain) for chain in chains)

    @property
    def block_sizes(self) -> list[int]:
        return [len(chain) for chain in self.chains]

    def __repr__(self) -> str:
        return 'JordanChains(%s, %d, blocks %s)' % (self.factor, self.multiplicity, self.block_sizes)


class _Field(Generic[T]):
    """The field of the entries, with the conversions."""

    def __init__(self, domain: Domain[T]) -> None:
        self.domain: Final[Domain[T]] = domain

    def element(self, e: Expr) -> T:
        raise NotImplementedError

    def expr(self, a: T) -> Expr:
        raise NotImplementedError

    def conjugate(self, a: T) -> T:
        """The complex conjugate, the symbols being real."""
        return a


class _RationalField(_Field[Fraction]):
    def __init__(self) -> None:
        super().__init__(QQ)
        self.lift: Final[_IntegerLift] = _IntegerLift()

    def element(self, e: Expr) -> Fraction:
        n, d = self.lift.fraction(e)
        return Fraction(n, d)

    def expr(self, a: Fraction) -> Expr:
        return number_from_fraction(a)


class _FunctionField(_Field[SparseFracElement[int]]):
    def __init__(self, lift: _PolynomialLift) -> None:
        field = SparseFractionField(lift.ring)
        super().__init__(field)
        self.field: Final[SparseFractionField[int]] = field
        self.lift: Final[_PolynomialLift] = lift

    def element(self, e: Expr) -> SparseFracElement[int]:
        n, d = self.lift.fraction(e)
        return self.field.new(n, d)

    def expr(self, a: SparseFracElement[int]) -> Expr:
        return self.lift.ratio(a.numer, a.denom)


class _GaussianField(_Field[_Radical]):
    """Rational functions of the symbols with ``I`` and square roots of
    rationals."""

    def __init__(self, lift: _ComplexLift) -> None:
        super().__init__(lift.field)
        self.lift: Final[_ComplexLift] = lift

    def element(self, e: Expr) -> _Radical:
        return self.lift.element(e)

    def expr(self, a: _Radical) -> Expr:
        return self.lift.expr(a)

    def conjugate(self, a: _Radical) -> _Radical:
        position = self.lift.position.get(-1)
        return a if position is None else self.lift.field.conjugate(a, position)


def _fresh_symbol(M: ExprMatrix) -> Symbol:
    used = {s.name for e in M.entries for s in e.free_symbols if isinstance(s, Symbol)}
    name, k = '_lambda', 0
    while name in used:
        name, k = '_lambda%d' % k, k + 1
    return Symbol(name)


def _field_nullspace(matrix: DomainMatrix[T]) -> list[list[T]]:
    """A basis of the null space over a field, one at the free
    coordinate."""
    F = matrix.domain
    reduced, pivots = matrix.rref()
    basis: list[list[T]] = []
    for free in range(matrix.ncols):
        if free in pivots:
            continue
        vector = [F.zero] * matrix.ncols
        vector[free] = F.one
        for i, column in enumerate(pivots):
            vector[column] = F.neg(reduced.rows[i][free])
        basis.append(vector)
    return basis


def _extension(field: _Field[T], factor: Expr, unknown: Symbol) -> QuotientField[T]:
    K = field.domain
    coefficients = [field.element(c) for c in _coefficients(factor, unknown)]
    leading = K.invert(coefficients[0])
    return QuotientField(K, [K.mul(c, leading) for c in coefficients])


def _shifted(field: _Field[T], F: QuotientField[T], M: ExprMatrix) -> DomainMatrix[list[T]]:
    """``M - t*I`` over ``F``, ``t`` its generator."""
    rows = [[F.sub(F.from_ground(field.element(e)), F.generator) if i == j else F.from_ground(field.element(e))
             for j, e in enumerate(M.row(i))] for i in range(M.rows)]
    return DomainMatrix(rows, F, M.cols)


def _column(field: _Field[T], vector: Sequence[list[T]], x: Expr) -> ExprMatrix:
    entries: list[Expr] = []
    for element in vector:
        degree = len(element) - 1
        entries.append(add(*[mul(field.expr(c), pow(x, Integer(degree - k))) for k, c in enumerate(element)]))
    return ExprMatrix.column(entries)


def _eigenspaces(field: _Field[T], M: ExprMatrix, factors: Sequence[tuple[Expr, int]], unknown: Symbol, x: Symbol
                 ) -> list[Eigenspace]:
    spaces: list[Eigenspace] = []
    for factor, multiplicity in factors:
        F = _extension(field, factor, unknown)
        basis = [_column(field, vector, x) for vector in _field_nullspace(_shifted(field, F, M))]
        spaces.append(Eigenspace(factor.subs(unknown, x), x, multiplicity, basis))
    return spaces


def _complete(F: QuotientField[T], size: int, known: Sequence[list[list[T]]], candidates: Sequence[list[list[T]]]
              ) -> list[list[list[T]]]:
    """The candidates which, added to the independent vectors ``known``,
    keep them independent (chosen greedily, in order)."""
    vectors = list(known) + list(candidates)
    if not vectors:
        return []
    columns = DomainMatrix([[v[i] for v in vectors] for i in range(size)], F, len(vectors))
    _, pivots = columns.rref()
    return [vectors[j] for j in pivots if j >= len(known)]


def _jordan_chains(field: _Field[T], M: ExprMatrix, factors: Sequence[tuple[Expr, int]], unknown: Symbol, x: Symbol
                   ) -> list[JordanChains]:
    n = M.rows
    result: list[JordanChains] = []
    for factor, multiplicity in factors:
        F = _extension(field, factor, unknown)
        N = _shifted(field, F, M)

        def image(v: Sequence[list[T]]) -> list[list[T]]:
            out: list[list[T]] = []
            for row in N.rows:
                total = F.zero
                for a, b in zip(row, v):
                    if a and b:
                        total = F.add(total, F.mul(a, b))
                out.append(total)
            return out

        # the kernels of the powers, up to the generalised eigenspace
        kernels: list[list[list[list[T]]]] = [[]]
        power = N
        while True:
            kernels.append(_field_nullspace(power))
            if len(kernels[-1]) == multiplicity:
                break
            if len(kernels[-1]) == len(kernels[-2]):
                raise ArithmeticError('the generalised eigenspace of %s has not the dimension %d' % (factor, multiplicity))
            power = power * N
        # from the longest chains down: at level k, the images of the level
        # above and new heads complete ker N**(k-1) to ker N**k
        chains: list[list[list[list[T]]]] = []
        level: list[list[list[T]]] = []
        heads: list[tuple[int, list[list[T]]]] = []
        for k in range(len(kernels) - 1, 0, -1):
            images = [image(v) for v in level]
            fresh = _complete(F, n, list(kernels[k - 1]) + images, kernels[k])
            heads.extend((k, v) for v in fresh)
            level = images + fresh
        for length, head in heads:
            chain = [head]
            for _ in range(length - 1):
                chain.append(image(chain[-1]))
            chain.reverse()
            chains.append(chain)
        result.append(JordanChains(factor.subs(unknown, x), x, multiplicity,
                                   [[_column(field, v, x) for v in chain] for chain in chains]))
    return result


def _characteristic_factors(M: ExprMatrix, unknown: Symbol) -> tuple[list[Expr], list[tuple[Expr, int]]]:
    """The kernels and the irreducible factors (in ``unknown``) of the
    characteristic polynomial with their multiplicities."""
    _square(M)
    kernels = _kernels([M])
    if not all(isinstance(k, Symbol) or _is_radical(k) for k in kernels):
        raise NotImplementedError('eigenvalues of a matrix whose entries are not rational functions of symbols, I and '
                                  'square roots of rationals')
    if M.rows == 0:
        return kernels, []
    polynomial = charpoly(M, unknown)
    symbols = [k for k in kernels if isinstance(k, Symbol)]
    numerator = expand(polynomial.together().as_numer_denom()[0]) if kernels else polynomial
    remaining = _kernels([ExprMatrix(1, 1, [numerator])])
    if any(_is_radical(k) and not isinstance(k, ImaginaryUnit) for k in remaining):
        raise NotImplementedError('the characteristic polynomial %s has irrational coefficients: its factorisation over '
                                  'the field they generate is needed' % (polynomial.subs(unknown, Symbol('x')),))
    if any(isinstance(k, ImaginaryUnit) for k in remaining):
        # over QQ(i): Trager's algorithm on the numerator
        ring = PolyRing(QQ_I, [unknown.name] + [k.name for k in symbols])
        _, gaussian = gaussian_ext_factor(ring.from_expr(numerator))
        return kernels, [(g.to_expr(), k) for g, k in gaussian if g.to_expr().has(unknown)]
    _, listed = factor_list(numerator, unknown, *symbols)
    factors: list[tuple[Expr, int]] = []
    for factor, exponent in listed:
        if not factor.has(unknown):
            continue
        if not isinstance(exponent, Integer):
            raise TypeError('a multiplicity which is not an integer: %s' % (exponent,))
        factors.append((factor, exponent.p))
    return kernels, factors


def eigenspaces(M: ExprMatrix, x: Symbol) -> list[Eigenspace]:
    """The eigenvalues and eigenvectors, grouped by the irreducible
    factors of the characteristic polynomial (in ``x``), in the order of
    :func:`~sympy_extras.polys.polytools.factor_list`.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.eigen import eigenspaces
    >>> x = Symbol('x')
    >>> C = ExprMatrix.from_rows([[0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
    >>> [space] = eigenspaces(C, x)
    >>> space.factor, space.multiplicity, space.values() is None
    (x**5 - x - 1, 1, True)
    >>> space.basis
    (ExprMatrix([[x**4 - 1], [x**3], [x**2], [x], [1]]),)

    For a matrix with floats, the exact structure of the matrix of the
    rational numbers they store.
    """
    if float_precision([M]) is not None:
        return eigenspaces(exact_matrix(M), x)
    unknown = _fresh_symbol(M)
    kernels, factors = _characteristic_factors(M, unknown)
    gaussian = _complex_lift(kernels)
    if gaussian is not None:
        return _eigenspaces(_GaussianField(gaussian), M, factors, unknown, x)
    if kernels:
        return _eigenspaces(_FunctionField(_PolynomialLift(kernels)), M, factors, unknown, x)
    return _eigenspaces(_RationalField(), M, factors, unknown, x)


def jordan_chains(M: ExprMatrix, x: Symbol) -> list[JordanChains]:
    """The Jordan structure, grouped by the irreducible factors of the
    characteristic polynomial (in ``x``): the chains of generalised
    eigenvectors, in terms of the eigenvalue. Does not need the roots of
    the factors.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.eigen import jordan_chains
    >>> x = Symbol('x')
    >>> M = ExprMatrix.from_rows([[3, 1, 0], [-1, 1, 0], [0, 0, 2]])
    >>> [blocks] = jordan_chains(M, x)
    >>> blocks, blocks.chains
    (JordanChains(x - 2, 3, blocks [2, 1]), ((ExprMatrix([[1], [-1], [0]]), ExprMatrix([[1], [0], [0]])), (ExprMatrix([[0], [0], [1]]),)))

    For a matrix with floats, the exact structure of the matrix of the
    rational numbers they store.
    """
    if float_precision([M]) is not None:
        return jordan_chains(exact_matrix(M), x)
    unknown = _fresh_symbol(M)
    kernels, factors = _characteristic_factors(M, unknown)
    gaussian = _complex_lift(kernels)
    if gaussian is not None:
        return _jordan_chains(_GaussianField(gaussian), M, factors, unknown, x)
    if kernels:
        return _jordan_chains(_FunctionField(_PolynomialLift(kernels)), M, factors, unknown, x)
    return _jordan_chains(_RationalField(), M, factors, unknown, x)


def jordan_form(M: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix]:
    """``(P, J)`` with ``J`` the Jordan normal form and ``M P = P J``,
    ``P`` invertible; ``NotImplementedError`` when some eigenvalues are
    not expressed in radicals (see :func:`jordan_chains`).

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.eigen import jordan_form
    >>> jordan_form(ExprMatrix.from_rows([[3, 1, 0], [-1, 1, 0], [0, 0, 2]]))
    (ExprMatrix([[1, 1, 0], [-1, 0, 0], [0, 0, 1]]), ExprMatrix([[2, 1, 0], [0, 2, 0], [0, 0, 2]]))

    With floats in the entries the structure is the exact one of the
    numbers stored, and ``P`` and ``J`` are written with floats:

    >>> from sympy_extras.core import Float
    >>> jordan_form(ExprMatrix.from_rows([[Float(1.0), Float(1.0)], [0, Float(1.0)]]))[1]
    ExprMatrix([[1.00000000000000, 1.00000000000000], [0, 1.00000000000000]])
    """
    digits = float_precision([M])
    if digits is not None:
        P, J = _jordan_form(exact_matrix(M), True)
        return approximate_matrix(P, digits), approximate_matrix(J, digits)
    return _jordan_form(M, False)


def _jordan_form(M: ExprMatrix, numerical: bool) -> tuple[ExprMatrix, ExprMatrix]:
    n = _square(M)
    blocks: list[tuple[Expr, list[ExprMatrix]]] = []
    for group in jordan_chains(M, _fresh_symbol(M)):
        found = _isolated_roots(group.factor, group.symbol) if numerical else _roots_in_radicals(group.factor, group.symbol)
        if found is None:
            raise NotImplementedError('the roots of %s are not expressed in radicals: see jordan_chains' % (group.factor,))
        for value in found:
            for vectors in group.chains:
                blocks.append((value, [v.subs(group.symbol, value) for v in vectors]))
    if numerical and all(not value.free_symbols for value, _ in blocks):
        blocks.sort(key=lambda block: (_numerical_order(block[0]), -len(block[1])))
    else:
        blocks.sort(key=lambda block: (default_sort_key(block[0]), -len(block[1])))
    columns: list[ExprMatrix] = []
    zero: Expr = Integer(0)
    one: Expr = Integer(1)
    J = [[zero] * n for _ in range(n)]
    for value, block in blocks:
        for position, vector in enumerate(block):
            index = len(columns)
            J[index][index] = value
            if position:
                J[index - 1][index] = one
            columns.append(vector)
    P = ExprMatrix(n, n, [columns[j][i, 0] for i in range(n) for j in range(n)])
    return P, ExprMatrix(n, n, [e for row in J for e in row])


def _explicit(M: ExprMatrix, numerical: bool = False) -> list[tuple[Expr, Eigenspace]]:
    """The eigenvalues with their spaces; with ``numerical`` (the exact
    matrix of a matrix with floats) as exact numbers to be evaluated, in
    numerical order."""
    pairs: list[tuple[Expr, Eigenspace]] = []
    for space in eigenspaces(M, _fresh_symbol(M)):
        values = space.isolated_values() if numerical else space.values()
        if values is None:
            raise NotImplementedError('the roots of %s are not expressed in radicals: see eigenspaces' % (space.factor,))
        pairs.extend((value, space) for value in values)
    if numerical and all(not value.free_symbols for value, _ in pairs):
        pairs.sort(key=lambda pair: _numerical_order(pair[0]))
    else:
        pairs.sort(key=lambda pair: default_sort_key(pair[0]))
    return pairs


def eigenvalues(M: ExprMatrix) -> dict[Expr, int]:
    """The eigenvalues with their algebraic multiplicities;
    ``NotImplementedError`` when some are not expressed in radicals (see
    :func:`eigenspaces`).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.eigen import eigenvalues
    >>> a, b = Symbol('a'), Symbol('b')
    >>> eigenvalues(ExprMatrix.from_rows([[a, b], [b, a]])), eigenvalues(ExprMatrix.from_rows([[a, 1], [0, a]]))
    ({a - b: 1, a + b: 1}, {a: 2})

    With floats in the entries, the roots of the exact characteristic
    polynomial (of any degree) as floats, with their exact
    multiplicities; eigenvalues which differ by less than the precision
    are one key, with the sum of the multiplicities.

    >>> from sympy_extras.core import Float
    >>> eigenvalues(ExprMatrix.from_rows([[Float(1.0), Float(1.0)], [0, Float(1.0)]]))
    {1.00000000000000: 2}
    """
    digits = float_precision([M])
    if digits is not None:
        counted: dict[Expr, int] = {}
        for value, space in _explicit(exact_matrix(M), True):
            key = approximate(value, digits)
            counted[key] = counted.get(key, 0) + space.multiplicity
        return counted
    return {value: space.multiplicity for value, space in _explicit(M)}


def eigenvectors(M: ExprMatrix) -> list[tuple[Expr, int, list[ExprMatrix]]]:
    """``(eigenvalue, algebraic multiplicity, basis of the eigenspace)``
    for every eigenvalue (SymPy's ``eigenvects``).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.eigen import eigenvectors
    >>> a, b = Symbol('a'), Symbol('b')
    >>> eigenvectors(ExprMatrix.from_rows([[a, b], [b, a]]))
    [(a - b, 1, [ExprMatrix([[-1], [1]])]), (a + b, 1, [ExprMatrix([[1], [1]])])]

    With floats in the entries the eigenspaces are those of the numbers
    stored: a defective matrix has fewer vectors than the multiplicity.

    >>> from sympy_extras.core import Float
    >>> eigenvectors(ExprMatrix.from_rows([[Float(1.0), Float(1.0)], [0, Float(1.0)]]))
    [(1.00000000000000, 2, [ExprMatrix([[1.00000000000000], [0]])])]
    """
    digits = float_precision([M])
    if digits is not None:
        return [(approximate(value, digits), space.multiplicity,
                 [approximate_matrix(v, digits) for v in space.vectors(value)])
                for value, space in _explicit(exact_matrix(M), True)]
    return [(value, space.multiplicity, space.vectors(value)) for value, space in _explicit(M)]


def is_diagonalizable(M: ExprMatrix) -> bool:
    """Whether the matrix is diagonalisable over the algebraic closure
    (for generic values of its symbols): every eigenvalue has as many
    independent eigenvectors as its multiplicity. Decided without solving
    the characteristic polynomial.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.eigen import is_diagonalizable
    >>> is_diagonalizable(ExprMatrix.from_rows([[0, 1], [-1, 0]])), is_diagonalizable(ExprMatrix.from_rows([[1, 1], [0, 1]]))
    (True, False)
    """
    return all(space.geometric_multiplicity == space.multiplicity for space in eigenspaces(M, _fresh_symbol(M)))


def diagonalize(M: ExprMatrix) -> Optional[tuple[ExprMatrix, ExprMatrix]]:
    """``(P, D)`` with ``D`` diagonal and ``M P = P D``, ``P`` invertible,
    or ``None`` when the matrix is not diagonalisable.

    >>> from sympy_extras.matrices import ExprMatrix
    >>> from sympy_extras.matrices.eigen import diagonalize
    >>> diagonalize(ExprMatrix.from_rows([[4, 1], [2, 3]]))
    (ExprMatrix([[-1/2, 1], [1, 1]]), ExprMatrix([[2, 0], [0, 5]]))
    >>> diagonalize(ExprMatrix.from_rows([[1, 1], [0, 1]])) is None
    True

    With floats in the entries, whether the matrix is diagonalisable is
    decided exactly for the numbers stored.
    """
    digits = float_precision([M])
    if digits is not None:
        exact = _diagonalize(exact_matrix(M), True)
        if exact is None:
            return None
        return approximate_matrix(exact[0], digits), approximate_matrix(exact[1], digits)
    return _diagonalize(M, False)


def _diagonalize(M: ExprMatrix, numerical: bool) -> Optional[tuple[ExprMatrix, ExprMatrix]]:
    n = _square(M)
    columns: list[ExprMatrix] = []
    diagonal: list[Expr] = []
    for value, space in _explicit(M, numerical):
        if space.geometric_multiplicity != space.multiplicity:
            return None
        for vector in space.vectors(value):
            columns.append(vector)
            diagonal.append(value)
    zero: Expr = Integer(0)
    P = ExprMatrix(n, n, [columns[j][i, 0] for i in range(n) for j in range(n)])
    D = ExprMatrix(n, n, [diagonal[i] if i == j else zero for i in range(n) for j in range(n)])
    return P, D
