"""Typed coefficient domains for the dense polynomial algorithms:
``ZZ``, ``QQ``, ``GF(p)``, polynomial rings and fraction fields.

A polynomial ring over a polynomial ring is the dense representation of
polynomials in several variables, so that every algorithm written for a
``Domain[T]`` is a multivariate algorithm.
"""
from .domain import Domain
from .finitefield import FiniteField, GF
from .fractionfield import FracElement, FractionField
from .integerring import IntegerRing, ZZ
from .polynomialring import PolynomialRing
from .rationalfield import RationalField, QQ

__all__ = ['Domain', 'IntegerRing', 'ZZ', 'RationalField', 'QQ', 'FiniteField', 'GF', 'PolynomialRing',
           'FractionField', 'FracElement']
