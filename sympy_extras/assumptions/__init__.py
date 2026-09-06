"""Assumptions written as mathematical statements.

An alternative front end to the assumptions of SymPy, modelled on the user
interface of Mathematica but with Python names. Instead of predicates like
``Q.positive(x)`` the assumptions are ordinary statements:

* ``x > 0``, ``x**2 + y**2 <= 1``, ``Eq(x, y)`` and ``Ne(x, 0)`` for
  ``Q.positive(x)`` and the like (an inequality states that both sides
  are real numbers, as in Mathematica);
* ``element(x, S.Integers)``, that is ``Contains(x, S.Integers)``, for
  ``Q.integer(x)``, and likewise ``S.Naturals``, ``S.Rationals``,
  ``S.Reals``, intervals, finite sets and their unions, intersections
  and complements;
* ``&``, ``|``, ``~`` or ``And``, ``Or``, ``Not``, ``Implies``,
  ``Equivalent``, ``Xor`` to combine them;
* ``ForAll(x, ...)`` and ``Exists(x, ...)`` for quantified statements.

The functions are

* :func:`ask` -- the truth value of a statement under assumptions;
* :func:`refine` and :func:`simplify` -- ``Refine`` and ``Simplify`` with
  assumptions;
* :func:`assuming` and :data:`global_assumptions` -- ``Assuming`` and
  ``$Assumptions``;
* :func:`resolve` -- ``Resolve``, quantifier elimination over the reals;
* :func:`satisfiable`, :func:`tautology` and :func:`find_instance` --
  ``SatisfiableQ``, ``TautologyQ`` and ``FindInstance``.

Two backends are used: the assumptions system of SymPy (:func:`sympy.ask`,
:func:`sympy.refine` and its SAT solver), which receives the predicates
implied by the assumptions, and the cylindrical algebraic decomposition of
:mod:`sympy_extras.polys.cad`, which decides exactly every statement which
is a Boolean combination of polynomial relations between real variables.

Examples
========

>>> from sympy import Abs, sqrt, S, Eq
>>> from sympy.abc import x, y, n
>>> from sympy_extras.assumptions import ask, refine, element, resolve, ForAll, Exists
>>> ask(x**2 - 2*x + 1 >= 0, x > 0)
True
>>> refine(Abs(x - 1) + sqrt(x**2), x > 2)
2*x - 1
>>> ask(element(n**2 + n, S.Integers), element(n, S.Integers))
True
>>> resolve(ForAll(x, Exists(y, Eq(y**2, x))))
False
>>> resolve(Exists(y, Eq(y**2, x) & (y > 1)))
x > 1
"""
from __future__ import annotations

from .facts import element, Facts
from .quantifiers import ForAll, Exists, Quantifier, prenex
from .context import global_assumptions, assuming, AssumptionsContext
from .ask import ask
from .refine import refine, simplify
from .resolve import resolve
from .sat import satisfiable, tautology, find_instance

__all__ = ['element', 'Facts', 'ForAll', 'Exists', 'Quantifier', 'prenex',
    'global_assumptions', 'assuming', 'AssumptionsContext', 'ask', 'refine',
    'simplify', 'resolve', 'satisfiable', 'tautology', 'find_instance']
