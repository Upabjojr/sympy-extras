"""Definite integration extending :mod:`sympy.integrals`."""
from __future__ import annotations

from .conditions import ConditionalValue
from .mellin import GammaQuotient, MellinTransform, mellin_transform, mellin_kernel
from .slater import slater_expansion, mellin_barnes, MeijerG
from .marichev import mellin_integrate
from .definite import definite_integral, conditional_integral, verify_numerically
from .regions import IntegralByRanges, integrate_by_ranges
from .residues import residue_integral

__all__ = ['ConditionalValue', 'GammaQuotient', 'MellinTransform', 'mellin_transform', 'mellin_kernel',
    'slater_expansion', 'mellin_barnes', 'MeijerG', 'mellin_integrate', 'definite_integral',
    'conditional_integral', 'verify_numerically', 'IntegralByRanges', 'integrate_by_ranges', 'residue_integral']
