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

The functions are, by module (the package exports nothing, so that
importing the native modules does not load the ones still written on
SymPy):

* :func:`~sympy_extras.assumptions.ask.ask` -- the truth value of a
  statement under assumptions;
* :func:`~sympy_extras.assumptions.refine.refine` and
  :func:`~sympy_extras.assumptions.refine.simplify` -- ``Refine`` and
  ``Simplify`` with assumptions;
* :func:`~sympy_extras.assumptions.context_sympy.assuming` and
  :data:`~sympy_extras.assumptions.context_sympy.global_assumptions` --
  ``Assuming`` and ``$Assumptions``;
* :func:`~sympy_extras.assumptions.resolve.resolve` -- ``Resolve``,
  quantifier elimination over the reals;
* :func:`~sympy_extras.assumptions.sat.satisfiable`, ``tautology`` and
  ``find_instance`` -- ``SatisfiableQ``, ``TautologyQ`` and ``FindInstance``;
* :func:`~sympy_extras.assumptions.solve.solve` -- ``Solve`` with
  assumptions and a domain;
* :func:`~sympy_extras.assumptions.limits.limit`, ``limit_seq`` and
  ``series`` -- limits and expansions with the parameters carrying the
  assumptions;
* :func:`~sympy_extras.assumptions.facts_sympy.element`,
  :class:`~sympy_extras.assumptions.facts_sympy.Facts`,
  :class:`~sympy_extras.assumptions.quantifiers_sympy.ForAll`,
  :class:`~sympy_extras.assumptions.quantifiers_sympy.Exists` and
  :func:`~sympy_extras.assumptions.quantifiers_sympy.prenex` -- the
  building blocks.

The leaves :mod:`~sympy_extras.assumptions.facts`,
:mod:`~sympy_extras.assumptions.quantifiers`,
:mod:`~sympy_extras.assumptions.context` and
:mod:`~sympy_extras.assumptions.bounds` are written on the core (no SymPy);
the modules ``facts_sympy``, ``quantifiers_sympy``, ``context_sympy`` and
``bounds_sympy`` are their SymPy faces, used by the modules above, which
are still written on SymPy objects.

Two backends are used: the assumptions system of SymPy (:func:`sympy.ask`,
:func:`sympy.refine` and its SAT solver), which receives the predicates
implied by the assumptions, and the cylindrical algebraic decomposition of
:mod:`sympy_extras.polys.cad`, which decides exactly every statement which
is a Boolean combination of polynomial relations between real variables.

Examples
========

>>> from sympy import Abs, sqrt, S, Eq
>>> from sympy.abc import x, y, n
>>> from sympy_extras.assumptions.ask import ask
>>> from sympy_extras.assumptions.refine import refine
>>> from sympy_extras.assumptions.resolve import resolve
>>> from sympy_extras.assumptions.facts_sympy import element
>>> from sympy_extras.assumptions.quantifiers_sympy import ForAll, Exists
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

__all__: list[str] = []
