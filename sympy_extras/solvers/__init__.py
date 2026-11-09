"""Solvers of differential and algebraic equations extending
:mod:`sympy.solvers`: Lie point symmetries of ODEs and PDEs, similarity
reductions and the solution of ODEs through their symmetries; linear
Diophantine systems, nonnegative integer solutions and Presburger
arithmetic; transcendental equations reduced to polynomial ones; polynomial
systems with the cases of their parameters; real roots
of transcendental functions isolated exactly; linear differential-algebraic
equations with constant coefficients; Thue equations; integrating factors,
linearisation and Abel invariants for ordinary differential equations;
factorisation of linear differential operators (Beke's algorithm); linear
systems with rational coefficients (cyclic vectors, Barkatou's rational
solutions, variation of constants); Darboux polynomials and the
Prelle–Singer procedure for first order equations ``y' = P/Q``.

The package exports nothing: the modules are imported by their paths
(``from sympy_extras.solvers.ode import dsolve_lie``), so that importing
the native ones (:mod:`~sympy_extras.solvers.recurr`,
:mod:`~sympy_extras.solvers.linsolve`) does not load the modules still
written on SymPy."""
from __future__ import annotations

__all__: list[str] = []
