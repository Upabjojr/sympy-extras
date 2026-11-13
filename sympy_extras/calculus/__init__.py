"""Calculus on the typed core (SymPy's ``sympy.calculus``): the period of a
function (``periodicity``, ``lcim``). The other modules are imported by
their paths: :mod:`sympy_extras.calculus.optimization` (exact global
optimisation of polynomials under polynomial constraints) and
:mod:`sympy_extras.calculus.interpolation`."""
from .util import lcim, periodicity

__all__ = ['periodicity', 'lcim']
