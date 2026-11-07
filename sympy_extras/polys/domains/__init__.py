"""Typed coefficient domains for the dense polynomial algorithms.

``ZZ``, ``QQ`` and ``GF(p)`` are exported here; the domains built on the
dense polynomial code (:class:`~sympy_extras.polys.domains.polynomialring.PolynomialRing`,
:class:`~sympy_extras.polys.domains.fractionfield.FractionField`,
:class:`~sympy_extras.polys.domains.algebraicfield.AlgebraicField`) are
imported from their own modules, since the dense modules import
:class:`~sympy_extras.polys.domains.domain.Domain` from this package.

A polynomial ring over a polynomial ring is the dense representation of
polynomials in several variables, so that every algorithm written for a
``Domain[T]`` is a multivariate algorithm.
"""
from .domain import Domain
from .finitefield import FiniteField, GF
from .integerring import IntegerRing, ZZ
from .rationalfield import RationalField, QQ

__all__ = ['Domain', 'IntegerRing', 'ZZ', 'RationalField', 'QQ', 'FiniteField', 'GF']
