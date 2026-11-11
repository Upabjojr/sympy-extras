"""The SymPy face of :mod:`sympy_extras.polys.hermite`: real solutions of
zero-dimensional systems counted by Hermite's quadratic form, and sign
determination.

The functions take SymPy expressions, ``Poly`` objects, symbols and the
ideals of :mod:`sympy_extras.polys.ideals_sympy` (over ``QQ``), convert
them to the core (:mod:`sympy_extras.interop`) and to the native ideals
(:func:`~sympy_extras.polys.ideals_sympy.native_ideal`), run the native
functions and convert the answers back: integers, sign conditions, SymPy
matrices of rationals. Nothing of the computation stays on SymPy. The
quotient algebra :class:`_Algebra` of a face ideal, with its vectors over
SymPy's ``QQ``, is what :mod:`sympy_extras.polys.rur` builds on.

Examples
========

>>> from sympy.abc import x, y
>>> from sympy_extras.polys.hermite_sympy import count_real_solutions, sign_determination
>>> count_real_solutions([x**2 + y**2 - 1, x - y], x, y)
2
>>> sign_determination([x**2 - 1, y**2 - 1], [x + y, x], x, y)
{(-1, -1): 1, (0, -1): 1, (0, 1): 1, (1, 1): 1}
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence, Union

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.matrices.dense import MutableDenseMatrix
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly
from sympy.polys.rings import PolyElement as SymPyPolyElement

from sympy_extras._typing import DomainElement, Truth, as_boolean, as_expr
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import hermite as native
from .domains.rationalfield import QQ as CoreQQ
from .hermite import SignCondition, SignCounts
from .ideals_sympy import Ideal, native_ideal
from .rings import PolyElement

__all__ = ['SignCounts', 'System', 'Vector', 'Rows', 'SignCondition', 'hermite_matrix', 'tarski_query',
           'count_real_solutions', 'count_complex_solutions', 'real_sign_counts', 'sign_determination',
           'decide_zero_dimensional']

#: a system: an ideal, or the polynomials which generate it
System = Union[Ideal, Sequence[Union[Expr, Poly]]]
#: a vector of rationals of SymPy's ``QQ``: coordinates in the basis of
#: standard monomials, or a linear form on the quotient algebra
Vector = list[DomainElement]
#: a square matrix of rationals of SymPy's ``QQ``, as a list of rows
Rows = list[list[DomainElement]]
#: a polynomial of the face
Polynomial = Union[Expr, Poly, int]


def _fraction(c: DomainElement) -> Fraction:
    return Fraction(int(QQ.numer(c)), int(QQ.denom(c)))


def _vector_out(v: native.Vector) -> Vector:
    return [QQ(c.numerator, c.denominator) for c in v]


def _vector_in(v: Vector) -> native.Vector:
    return [_fraction(c) for c in v]


def _rows_out(rows: native.Rows) -> Rows:
    return [_vector_out(row) for row in rows]


def _rows_in(rows: Rows) -> native.Rows:
    return [_vector_in(row) for row in rows]


def _core_expr(e: Union[Expr, Poly]) -> CoreExpr:
    converted = from_sympy(as_expr(e.as_expr()) if isinstance(e, Poly) else as_expr(sympify(e)))
    if not isinstance(converted, CoreExpr):
        raise TypeError("%s is not an expression" % (e,))
    return converted


def _element(g: Polynomial) -> native.Element:
    """A polynomial of the face as one of the native algebra."""
    if isinstance(g, int):
        return g
    return _core_expr(g)


def _core_symbol(s: Symbol) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


class _Algebra:
    """The quotient algebra of a zero-dimensional ideal of the face over
    ``QQ`` (the native :class:`sympy_extras.polys.hermite._Algebra`), with
    its vectors and matrices over SymPy's ``QQ``; its elements are those
    of the ring of the native ideal."""

    def __init__(self, ideal: Ideal) -> None:
        if ideal.domain != QQ:
            raise NotImplementedError("real zeros are counted for ideals over QQ, not %s" % ideal.domain)
        self.ideal: Ideal = ideal
        self.native: native._Algebra = native._Algebra(native_ideal(ideal, CoreQQ))
        self.empty: bool = self.native.empty
        self.index: dict[tuple[int, ...], int] = self.native.index
        R = ideal.ring()
        one = R.domain.one
        #: the standard monomials, as elements of the ring of the face
        self.basis: list[SymPyPolyElement] = [R.from_dict({m: one}) for m in self.native.monomials]
        self.trace: Vector = _vector_out(self.native.trace)

    def element(self, g: Polynomial) -> PolyElement[Fraction]:
        """``g`` as an element of the ring of the native ideal."""
        return self.native.element(_element(g))

    def coordinates(self, f: PolyElement[Fraction]) -> Vector:
        """The coordinates of the normal form of ``f``."""
        return _vector_out(self.native.coordinates(f))

    def transposed_multiplication(self, g: PolyElement[Fraction]) -> Rows:
        """The transpose of the matrix of multiplication by ``g``."""
        return _rows_out(self.native.transposed_multiplication(g))

    def compose(self, form: Vector, transposed: Rows) -> Vector:
        """The linear form ``h -> form(g h)``, for ``transposed`` the
        transposed multiplication matrix of ``g``."""
        return _vector_out(self.native.compose(_vector_in(form), _rows_in(transposed)))

    def hermite(self, form: Vector) -> Rows:
        """The matrix of ``(p, q) -> form(p q)`` in the standard monomials."""
        return _rows_out(self.native.hermite(_vector_in(form)))


def _ideal(system: System, gens: Sequence[Symbol]) -> Ideal:
    return system if isinstance(system, Ideal) else Ideal(list(system), *gens)


def _algebra(system: System, gens: Sequence[Symbol]) -> _Algebra:
    return _Algebra(_ideal(system, gens))


def _native_system(system: System, gens: Sequence[Symbol]) -> native.System:
    """The native ideal of a system of the face (over ``QQ``)."""
    ideal = _ideal(system, gens)
    if ideal.domain != QQ:
        raise NotImplementedError("real zeros are counted for ideals over QQ, not %s" % ideal.domain)
    return native_ideal(ideal, CoreQQ)


def hermite_matrix(system: System, g: Polynomial = 1, *gens: Symbol) -> MutableDenseMatrix:
    r""":func:`~sympy_extras.polys.hermite.hermite_matrix` on SymPy
    objects: the matrix of Hermite's quadratic form $(p, q) \mapsto
    \operatorname{Tr}(g p q)$ in the basis of the standard monomials.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy_extras.polys.hermite_sympy import hermite_matrix
    >>> hermite_matrix([x**2 - 2], 1, x)
    Matrix([[2, 0], [0, 4]])
    >>> hermite_matrix([x**2 - 2], x, x)
    Matrix([[0, 4], [4, 0]])
    """
    found = native.hermite_matrix(_native_system(system, gens), _element(g))
    return MutableDenseMatrix(found.rows, found.cols, [to_sympy(e) for e in found.entries])


def tarski_query(system: System, g: Polynomial = 1, *gens: Symbol) -> int:
    """:func:`~sympy_extras.polys.hermite.tarski_query` on SymPy objects:
    the number of distinct real zeros at which ``g > 0`` minus the number
    at which ``g < 0``.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite_sympy import tarski_query
    >>> tarski_query([x**2 + y**2 - 1, x - 2*y], x, x, y)
    0
    >>> tarski_query([x**3 - x, y - x**2], 2*y - 1, x, y)
    1
    """
    return native.tarski_query(_native_system(system, gens), _element(g))


def count_real_solutions(system: System, *gens: Symbol) -> int:
    """:func:`~sympy_extras.polys.hermite.count_real_solutions` on SymPy
    objects: the number of distinct real solutions of a zero-dimensional
    system with rational coefficients.

    Raises
    ======

    NotImplementedError
        If the system has infinitely many complex solutions or its
        coefficients are not rational.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite_sympy import count_real_solutions
    >>> count_real_solutions([x**2 + y**2 - 1, x - y], x, y)
    2
    >>> count_real_solutions([(x - 1)**3*(x + 2), y**2], x, y)
    2
    """
    return native.count_real_solutions(_native_system(system, gens))


def count_complex_solutions(system: System, *gens: Symbol) -> int:
    """:func:`~sympy_extras.polys.hermite.count_complex_solutions` on SymPy
    objects: the number of distinct complex solutions.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite_sympy import count_complex_solutions
    >>> count_complex_solutions([x**2 + 1, (y - 1)**2], x, y)
    2
    """
    return native.count_complex_solutions(_native_system(system, gens))


def real_sign_counts(system: System, g: Polynomial, *gens: Symbol) -> SignCounts:
    """:func:`~sympy_extras.polys.hermite.real_sign_counts` on SymPy
    objects.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite_sympy import real_sign_counts
    >>> real_sign_counts([x**3 - x, y - x**2], x - y, x, y)
    SignCounts(zero=2, positive=0, negative=1)
    """
    return native.real_sign_counts(_native_system(system, gens), _element(g))


def sign_determination(system: System, polys: Sequence[Polynomial], *gens: Symbol) -> dict[SignCondition, int]:
    """:func:`~sympy_extras.polys.hermite.sign_determination` on SymPy
    objects: the realizable sign conditions of the polynomials on the real
    solutions, with the number of solutions realizing each.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite_sympy import sign_determination
    >>> sign_determination([x**2 - 1, y**2 - 1], [x + y, x], x, y)
    {(-1, -1): 1, (0, -1): 1, (0, 1): 1, (1, 1): 1}
    >>> sign_determination([x**2 + 1, y], [x], x, y)
    {}
    """
    return native.sign_determination(_native_system(system, gens), [_element(p) for p in polys])


def decide_zero_dimensional(formula: Union[Boolean, bool], gens: Sequence[Symbol],
                            max_dimension: int = 64) -> Truth:
    """:func:`~sympy_extras.polys.hermite.decide_zero_dimensional` on SymPy
    formulas: whether a conjunction of polynomial relations with rational
    coefficients whose equations have finitely many complex solutions has
    a real solution (``None`` when the formula is not of this form).

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.hermite_sympy import decide_zero_dimensional
    >>> decide_zero_dimensional(Eq(x**2 + y**2, 1) & Eq(x*y, 1), [x, y])
    False
    >>> decide_zero_dimensional(Eq(x**2 + y**2, 4) & Eq(x*y, 1) & (x > y), [x, y])
    True
    >>> decide_zero_dimensional(Eq(x**2 + y**2, 1) & (x > y), [x, y]) is None
    True
    """
    if isinstance(formula, bool):
        return formula
    converted = from_sympy(as_boolean(formula))
    if not isinstance(converted, CoreBoolean):
        return None
    return native.decide_zero_dimensional(converted, [_core_symbol(g) for g in gens], max_dimension)
