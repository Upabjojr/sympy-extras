"""Solvers of differential and algebraic equations extending
:mod:`sympy.solvers`: Lie point symmetries of ODEs and PDEs, similarity
reductions and the solution of ODEs through their symmetries; linear
Diophantine systems, nonnegative integer solutions and Presburger
arithmetic; transcendental equations reduced to polynomial ones."""
from .lie import JetSpace, Symmetry, symmetries, check_symmetry
from .pde import pde_symmetries, similarity_reduction, pdsolve_lie, Reduction
from .ode import (ode_symmetries, canonical_coordinates, reduce_order,
    dsolve_lie, solve_ode, ReducedODE)
from .integers import (hermite_normal_form_with_transform, linear_diophantine_system,
    hilbert_basis, minimal_nonnegative_solutions, cooper, presburger_quantifier_elimination,
    is_presburger)
from .transcendental import solve_transcendental, polynomialize

__all__ = ['JetSpace', 'Symmetry', 'symmetries', 'check_symmetry',
    'pde_symmetries', 'similarity_reduction', 'pdsolve_lie', 'Reduction',
    'ode_symmetries', 'canonical_coordinates', 'reduce_order', 'dsolve_lie',
    'solve_ode', 'ReducedODE',
    'hermite_normal_form_with_transform', 'linear_diophantine_system', 'hilbert_basis',
    'minimal_nonnegative_solutions', 'cooper', 'presburger_quantifier_elimination',
    'is_presburger', 'solve_transcendental', 'polynomialize']
