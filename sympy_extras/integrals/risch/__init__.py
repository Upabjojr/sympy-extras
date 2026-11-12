# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/risch.py (the subpackage) (SymPy 1.14 and Aaron Meurer's pull requests)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: the typed core and the polynomials of integrals.risch.tower.
# ---------------------------------------------------------------------
"""The transcendental Risch algorithm, ported from SymPy with Aaron
Meurer's unmerged extensions.

SymPy's :func:`sympy.integrals.risch.risch_integrate` implements the
transcendental case of the Risch algorithm (Bronstein, *Symbolic
Integration I*) for towers of exponentials and logarithms, with several
cases of the Risch differential equation and of its parametric version
left unimplemented (they raise ``NotImplementedError``). Aaron Meurer's
pull requests to SymPy complete them:

* sympy/sympy#30180 and sympy/sympy#30221 (*Risch: implement most
  remaining exp-log cases from Bronstein*): the cancellation cases of the
  Risch differential equation (``rde.py``), the parametric cancellation
  cases and the structure theorems (``prde.py``: ``prde_no_cancel_b_equal``,
  ``prde_cancel_liouvillian``, ``is_deriv_in_field``,
  ``parametric_log_deriv`` through the structure theorem), the termination
  guards of ``spde``, real arc-tangents from complex residues through
  Rioboo's ``log_to_real`` (``rationaltools.py``);
* sympy/sympy#30292 (*Hypertangent cases in the Risch algorithm*): the
  coupled differential system (``cde.py``, Bronstein chapter 8) and the
  hypertangent monomials (``integrate_hypertangent``, the tangent cases of
  the special denominators, degree bounds and cancellation), so that
  towers with ``tan`` and ``atan`` are integrated (``risch_integrate(tan(x)**5, x)``).

The modules ``risch.py``, ``rde.py``, ``prde.py``, ``cde.py``,
``rationaltools.py`` and ``polymatrix.py`` are ported from the files of the
branch ``risch-hypertangent`` at commit ``66bc72d6``, under SymPy's BSD
licence (``LICENSE-SymPy``), and rewritten on the typed core: the
polynomials of the tower are :class:`.tower.TowerPoly` over the field of
:class:`.tower.RationalFunction` elements, the matrices
:class:`.polymatrix.FieldMatrix`. Two further branches were not merged
in: ``risch-typing`` (sympy/sympy#30282, annotations written against
#30221, which conflict with the hypertangent changes) and
``risch-algebraic`` (sympy/sympy#30239, radicals as ``exp(log(x)/2)``
handled by the transcendental algorithm, which its author calls
experimental and which rewrites the same functions).
:mod:`.risch_sympy` is the face on SymPy objects (``risch_integrate``,
``risch_antiderivative``, ``is_nonelementary``).

The package exports nothing, so that importing its native modules does
not load SymPy: import the modules.

Examples
========

>>> from sympy import symbols, tan, exp, log
>>> from sympy_extras.integrals.risch.risch_sympy import risch_antiderivative, is_nonelementary
>>> x = symbols('x')
>>> risch_antiderivative(tan(x)**2, x)
-x + tan(x)
>>> risch_antiderivative(exp(x)*log(exp(x) + 1), x)
exp(x)*log(exp(x) + 1) - exp(x) + log(exp(x) + 1)
>>> risch_antiderivative(exp(x)/((exp(x) + 1)**2 + 1), x)
atan(exp(x) + 1)
>>> is_nonelementary(exp(x**2), x)
True
>>> risch_antiderivative(exp(x**2), x) is None
True

References
==========

.. [Bronstein] M. Bronstein, *Symbolic Integration I: Transcendental
   Functions*, 2nd edition, Springer, 2005.
.. [Meurer] A. Meurer, pull requests sympy/sympy#30180, #30221, #30292
   (2025-2026), https://github.com/sympy/sympy/pulls?q=author%3Aasmeurer+risch
"""
