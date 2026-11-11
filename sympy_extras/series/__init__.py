"""Series tools on the typed core (SymPy's ``sympy.series``): ``residue``;
Padé approximants, rational interpolation and the reversion of series
(:mod:`sympy_extras.series.approximants`)."""
from .approximants import pade, pade_from_coefficients, rational_interpolate, series_reversion, taylor_coefficients
from .residues import residue

__all__ = ['residue', 'pade', 'pade_from_coefficients', 'rational_interpolate', 'series_reversion', 'taylor_coefficients']
