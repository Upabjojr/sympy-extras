"""The SymPy face of the projection operators
(:mod:`sympy_extras.polys.cad.projection`).

The functions take and return SymPy's ``Poly`` over ``ZZ`` in SymPy
symbols, converted around the native functions: the coefficients go
through dictionaries of rational numbers, the generators through
:mod:`sympy_extras.interop`. The errors of the native polynomials come
out as SymPy's ``PolynomialError``. Nothing of the projection stays on
SymPy.

Examples
========

>>> from sympy import Poly
>>> from sympy.abc import x, y
>>> from sympy_extras.polys.cad.projection_sympy import projection_sets, mccallum_projection
>>> projection_sets([x**2 + y**2 - 1, x - y], [x, y])
[[Poly(x - 1, x, domain='ZZ'), Poly(x + 1, x, domain='ZZ'), Poly(2*x**2 - 1, x, domain='ZZ')], [Poly(x**2 + y**2 - 1, x, y, domain='ZZ'), Poly(x - y, x, y, domain='ZZ')]]
>>> mccallum_projection([Poly(x*y - 1, x, y), Poly(y - x, x, y)], y, equational=[Poly(y - x, x, y)])
[Poly(x - 1, x, domain='ZZ'), Poly(x + 1, x, domain='ZZ')]
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
from typing import Callable, Iterable, Iterator, Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import Integer, Rational
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ, ZZ
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly

from sympy_extras._typing_sympy import ExprLike, as_expr, as_symbol
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import Poly as CorePoly
from sympy_extras.polys.rings import Monomial

from . import projection as native

__all__ = ['squarefree_basis', 'mccallum_projection', 'hong_projection', 'projection_sets',
           'augmented_projection_sets', 'core_poly', 'sympy_poly', 'core_gens', 'sympy_errors']

#: the polynomial arguments
PolyLike = Union[ExprLike, Poly]

#: the conversions done so far, both ways (a polynomial converted back
#: comes out as the object it came from)
_to_core: dict[tuple[Poly, tuple[CoreExpr, ...]], CorePoly] = {}
_to_sympy: dict[tuple[CorePoly, tuple[Basic, ...]], Poly] = {}
_symbols: dict[Basic, CoreExpr] = {}
_KEPT = 20000


@contextmanager
def sympy_errors() -> Iterator[None]:
    """Raise the errors of the native polynomials as SymPy's
    ``PolynomialError``."""
    try:
        yield
    except BasePolynomialError as error:
        raise PolynomialError(str(error)) from error


def _core_symbol(s: Basic) -> CoreExpr:
    found = _symbols.get(s)
    if found is None:
        converted = from_sympy(s)
        if not isinstance(converted, CoreExpr):
            raise TypeError("%s is not an expression" % (s,))
        _symbols[s] = found = converted
    return found


def core_gens(gens: Iterable[Basic]) -> tuple[CoreExpr, ...]:
    """SymPy generators as expressions of the core."""
    return tuple(_core_symbol(g) for g in gens)


def _fraction(c: Basic) -> Fraction:
    value = as_expr(c)
    if not isinstance(value, Rational):
        raise PolynomialError("polynomial with rational coefficients expected, got the coefficient %s" % (value,))
    return Fraction(int(value.p), int(value.q))


def core_poly(p: Poly, gens: Optional[Sequence[CoreExpr]] = None) -> CorePoly:
    """The SymPy polynomial ``p`` over ``ZZ`` or ``QQ`` as a native
    polynomial, in ``gens`` (the conversion of its own generators by
    default)."""
    generators = core_gens(p.gens) if gens is None else tuple(gens)
    key = (p, generators)
    found = _to_core.get(key)
    if found is not None:
        return found
    if p.domain == ZZ or p.domain == QQ:
        terms: dict[Monomial, Union[int, Fraction]] = {tuple(m): _fraction(c) for m, c in p.as_dict().items()}
        result = CorePoly.from_dict(terms, *core_gens(p.gens))
        if result.gens != generators:
            result = CorePoly(result.as_expr(), *generators)
    else:
        with sympy_errors():
            result = CorePoly(native_expr(p.as_expr()), *generators)
    if len(_to_core) > _KEPT:
        _to_core.clear()
    _to_core[key] = result
    return result


def native_expr(e: Basic) -> CoreExpr:
    """A SymPy expression as an expression of the core."""
    converted = from_sympy(e)
    if not isinstance(converted, CoreExpr):
        raise TypeError("expected an expression, got %s" % (e,))
    return converted


def sympy_poly(p: CorePoly, gens: Sequence[Basic]) -> Poly:
    """The native polynomial ``p`` (over ``ZZ`` or ``QQ``) as a SymPy
    polynomial in the SymPy generators ``gens`` (those of ``p``,
    converted)."""
    generators = tuple(gens)
    key = (p, generators)
    found = _to_sympy.get(key)
    if found is not None:
        return found
    if p.is_integral:
        result = Poly.from_dict({m: Integer(int(c)) for m, c in p.rep.terms.items()}, *generators, domain=ZZ)
    else:
        result = Poly.from_dict({m: Rational(c.numerator, c.denominator) for m, c in p.rep.terms.items()},
                                *generators, domain=QQ)
    if len(_to_sympy) > _KEPT:
        _to_sympy.clear()
    _to_sympy[key] = result
    _to_core.setdefault((result, p.gens), p)
    return result


def _sympy_gens(p: CorePoly) -> tuple[Basic, ...]:
    return tuple(as_symbol(to_sympy(g)) for g in p.gens)


def sympy_polys(polys: Iterable[CorePoly]) -> list[Poly]:
    """Native polynomials as SymPy polynomials in their generators."""
    return [sympy_poly(p, _sympy_gens(p)) for p in polys]


def _core_input(p: PolyLike) -> native.PolyLike:
    if isinstance(p, Poly):
        if p.domain == ZZ or p.domain == QQ:
            return core_poly(p)
        return native_expr(p.as_expr())
    if isinstance(p, int):
        return p
    return native_expr(as_expr(p))


def _to_polys(polys: Iterable[PolyLike], gens: Sequence[Symbol]) -> list[Poly]:
    """:func:`~sympy_extras.polys.cad.projection.to_polys` on SymPy
    objects: polynomials over ZZ in ``gens``."""
    generators = tuple(gens)
    items = list(polys)
    if all(isinstance(p, Poly) and p.gens == generators and p.domain == ZZ for p in items):
        return [p for p in items if isinstance(p, Poly)]
    with sympy_errors():
        converted = native.to_polys([_core_input(p) for p in items], core_gens(generators))
    return [sympy_poly(p, generators) for p in converted]


def squarefree_basis(polys: Iterable[PolyLike], gens: Sequence[Symbol]) -> list[list[Poly]]:
    """:func:`~sympy_extras.polys.cad.projection.squarefree_basis` on
    SymPy objects.

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.projection_sympy import squarefree_basis
    >>> squarefree_basis([(x**2 - 1)*(x + y)**2, 6], [x, y])
    [[Poly(x - 1, x, y, domain='ZZ'), Poly(x + 1, x, y, domain='ZZ')], [Poly(x + y, x, y, domain='ZZ')]]
    """
    generators = tuple(gens)
    with sympy_errors():
        levels = native.squarefree_basis([_core_input(p) for p in polys], core_gens(generators))
    return [[sympy_poly(p, generators) for p in level] for level in levels]


def _main_projection(project: Callable[[list[CorePoly], CoreExpr, list[CorePoly]], list[CorePoly]],
                     polys: Iterable[Poly], x: Symbol, equational: Iterable[Poly]) -> list[Poly]:
    items = list(polys)
    if not items:
        return []
    with sympy_errors():
        projected = project([core_poly(p) for p in items], _core_symbol(x), [core_poly(p) for p in equational])
    return sympy_polys(projected)


def _mccallum(polys: list[CorePoly], x: CoreExpr, equational: list[CorePoly]) -> list[CorePoly]:
    return native.mccallum_projection(polys, x, equational)


def _hong(polys: list[CorePoly], x: CoreExpr, equational: list[CorePoly]) -> list[CorePoly]:
    return native.hong_projection(polys, x)


def mccallum_projection(polys: Iterable[Poly], x: Symbol, equational: Iterable[Poly] = ()) -> list[Poly]:
    """:func:`~sympy_extras.polys.cad.projection.mccallum_projection` on
    SymPy polynomials.

    >>> from sympy import Poly
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.projection_sympy import mccallum_projection
    >>> mccallum_projection([Poly(x**2 + y**2 - 1, x, y)], y)
    [Poly(x - 1, x, domain='ZZ'), Poly(x + 1, x, domain='ZZ')]
    """
    return _main_projection(_mccallum, polys, x, equational)


def hong_projection(polys: Iterable[Poly], x: Symbol) -> list[Poly]:
    """:func:`~sympy_extras.polys.cad.projection.hong_projection` on SymPy
    polynomials.

    >>> from sympy import Poly
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.projection_sympy import hong_projection
    >>> hong_projection([Poly(x*y - 1, x, y), Poly(y - x, x, y)], y)
    [Poly(x, x, domain='ZZ'), Poly(x - 1, x, domain='ZZ'), Poly(x + 1, x, domain='ZZ')]
    """
    return _main_projection(_hong, polys, x, ())


#: a projection operator on SymPy polynomials
Projection = Callable[[Iterable[Poly], Symbol], list[Poly]]

#: the projection operators by name
_PROJECTIONS: dict[str, Projection] = {
    'mccallum': mccallum_projection,
    'hong': hong_projection,
}


def projection_sets(polys: Iterable[PolyLike], gens: Sequence[Symbol], method: str = 'mccallum',
                    equational: Optional[PolyLike] = None) -> list[list[Poly]]:
    """:func:`~sympy_extras.polys.cad.projection.projection_sets` on SymPy
    objects."""
    generators = tuple(gens)
    with sympy_errors():
        result = native.projection_sets([_core_input(p) for p in polys], core_gens(generators), method,
                                        None if equational is None else _core_input(equational))
    return [[sympy_poly(p, generators[:k]) for p in level] for k, level in enumerate(result, start=1)]


def augmented_projection_sets(projection: Sequence[Sequence[Poly]], gens: Sequence[Symbol],
                              extra: Iterable[PolyLike], method: str = 'mccallum') -> list[list[Poly]]:
    """:func:`~sympy_extras.polys.cad.projection.augmented_projection_sets`
    on SymPy objects.

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.projection_sympy import projection_sets, augmented_projection_sets
    >>> augmented_projection_sets(projection_sets([x - y**2], [x, y]), [x, y], [y - 1])
    [[Poly(x, x, domain='ZZ'), Poly(x - 1, x, domain='ZZ')], [Poly(x - y**2, x, y, domain='ZZ'), Poly(y - 1, x, y, domain='ZZ')]]
    """
    generators = tuple(gens)
    with sympy_errors():
        result = native.augmented_projection_sets([[core_poly(p) for p in level] for level in projection],
                                                  core_gens(generators), [_core_input(p) for p in extra], method)
    return [[sympy_poly(p, generators[:k]) for p in level] for k, level in enumerate(result, start=1)]


def core_projection(projection: Sequence[Sequence[Poly]]) -> list[list[CorePoly]]:
    """Projection factor sets of SymPy polynomials as native ones."""
    return [[core_poly(p) for p in level] for level in projection]


def as_sympy_expr(e: CoreExpr) -> Expr:
    """An expression of the core as a SymPy expression."""
    return as_expr(to_sympy(e))
