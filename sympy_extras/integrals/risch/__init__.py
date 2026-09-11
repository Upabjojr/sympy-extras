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
``rationaltools.py`` and ``polymatrix.py`` are the files of the branch
``risch-hypertangent`` at commit ``66bc72d6`` with their imports changed,
under SymPy's BSD licence (``LICENSE-SymPy``). Two further branches were
not merged in: ``risch-typing`` (sympy/sympy#30282, annotations written
against #30221, which conflict with the hypertangent changes) and
``risch-algebraic`` (sympy/sympy#30239, radicals as ``exp(log(x)/2)``
handled by the transcendental algorithm, which its author calls
experimental and which rewrites the same functions). The ported code is
not typed to the standard of this package (see ``pyproject.toml``); this
module is the typed entry point.

Examples
========

>>> from sympy import symbols, tan, exp, log
>>> from sympy_extras.integrals.risch import risch_antiderivative, is_nonelementary
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
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.settings import settings
from .risch import NonElementaryIntegral, risch_integrate

__all__ = ['risch_antiderivative', 'is_nonelementary', 'NonElementaryIntegral']


def _run(f: Expr, x: Symbol) -> Optional[Expr]:
    """``risch_integrate`` under the time limit; ``None`` when it gives up
    (``NotImplementedError``, ``ValueError``, the time)."""
    return attempt(lambda: as_expr(risch_integrate(f, x)), settings.timeout)


def risch_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """An elementary antiderivative of ``f`` with respect to ``x`` by the
    transcendental Risch algorithm, or ``None`` when ``f`` has none (the
    algorithm proves it) or when the algorithm does not apply (an
    algebraic function, an unimplemented case, the time limit).

    Examples
    ========

    >>> from sympy import symbols, tan, exp, log, sin
    >>> from sympy_extras.integrals.risch import risch_antiderivative
    >>> x = symbols('x')
    >>> risch_antiderivative(tan(x), x)
    log(tan(x)**2 + 1)/2
    >>> risch_antiderivative(1/(x*(log(x)**2 + 1)), x)
    atan(log(x))
    >>> risch_antiderivative(sin(x)/x, x) is None
    True
    """
    f_ = as_expr(f)
    found = _run(f_, x)
    if found is None or found.has(Integral):
        return None
    return found


def is_nonelementary(f: ExprLike, x: Symbol) -> Optional[bool]:
    """Whether the Risch algorithm proves that ``f`` has no elementary
    antiderivative (``True``), finds one (``False``), or cannot tell
    (``None``: an unimplemented case, a non-transcendental tower, the
    time limit).

    Examples
    ========

    >>> from sympy import symbols, exp, log
    >>> from sympy_extras.integrals.risch import is_nonelementary
    >>> x = symbols('x')
    >>> is_nonelementary(exp(x)/x, x)
    True
    >>> is_nonelementary(exp(x)*x, x)
    False
    """
    found = _run(as_expr(f), x)
    if found is None:
        return None
    if found.has(NonElementaryIntegral):
        return True
    if found.has(Integral):
        return None
    return False
