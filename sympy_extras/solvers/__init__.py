"""Solvers of differential and algebraic equations extending
:mod:`sympy.solvers`: Lie point symmetries of ODEs and PDEs, similarity
reductions and the solution of ODEs through their symmetries; linear
Diophantine systems, nonnegative integer solutions and Presburger
arithmetic; transcendental equations reduced to polynomial ones; real roots
of transcendental functions isolated exactly; linear differential-algebraic
equations with constant coefficients; Thue equations; integrating factors,
linearisation and Abel invariants for ordinary differential equations."""
from .lie import JetSpace, Symmetry, symmetries, check_symmetry
from .pde import pde_symmetries, similarity_reduction, pdsolve_lie, Reduction
from .ode import (ode_symmetries, canonical_coordinates, reduce_order,
    dsolve_lie, solve_ode, ReducedODE)
from .integers import (hermite_normal_form_with_transform, linear_diophantine_system,
    hilbert_basis, minimal_nonnegative_solutions, cooper, presburger_quantifier_elimination,
    is_presburger)
from .transcendental import solve_transcendental, polynomialize
from .kovacic import liouvillian_solution, dsolve_kovacic, KovacicSolution
from .linear_ode import (LinearOperator, polynomial_solutions, rational_solutions,
    hyperexponential_solutions, reduce_order_linear, dsolve_linear)
from .charpit import complete_integral, check_complete_integral
from .special import special_solutions, bessel_solutions, whittaker_solutions, hypergeometric_solutions, whittaker_m
from .first_order import riccati_ode, chini_ode, abel_ode, lagrange_ode, dsolve_first_order
from .linear_systems import cyclic_vector, system_to_scalar, dsolve_linear_system, rational_system_solutions
from .dae import core_nilpotent_decomposition, dae_index, dsolve_dae, dae_matrices, check_dae, DAESolution
from .isolation import TranscendentalRoot, isolate_real_roots, real_roots_of
from .abel import (abel_invariants, abel_equivalence, abel_by_invariants, air_solution, particular_solution,
    AbelInvariants, AbelTransformation)
from .thue_equation import thue, units_of_order, elements_of_norm, ThueEquation
from .second_order import (integrating_factor_xy, integrating_factor_p, is_linearizable, linearize,
    rectify_symmetries, dsolve_second_order, FirstIntegral, Linearization)

__all__ = ['JetSpace', 'Symmetry', 'symmetries', 'check_symmetry',
    'pde_symmetries', 'similarity_reduction', 'pdsolve_lie', 'Reduction',
    'ode_symmetries', 'canonical_coordinates', 'reduce_order', 'dsolve_lie',
    'solve_ode', 'ReducedODE',
    'hermite_normal_form_with_transform', 'linear_diophantine_system', 'hilbert_basis',
    'minimal_nonnegative_solutions', 'cooper', 'presburger_quantifier_elimination',
    'is_presburger', 'solve_transcendental', 'polynomialize',
    'liouvillian_solution', 'dsolve_kovacic', 'KovacicSolution', 'LinearOperator', 'polynomial_solutions',
    'rational_solutions', 'hyperexponential_solutions', 'reduce_order_linear', 'dsolve_linear',
    'complete_integral', 'check_complete_integral',
    'special_solutions', 'bessel_solutions', 'whittaker_solutions', 'hypergeometric_solutions',
    'whittaker_m', 'riccati_ode', 'chini_ode', 'abel_ode', 'lagrange_ode', 'dsolve_first_order',
    'cyclic_vector', 'system_to_scalar', 'dsolve_linear_system', 'rational_system_solutions',
    'core_nilpotent_decomposition', 'dae_index', 'dsolve_dae', 'dae_matrices', 'check_dae', 'DAESolution',
    'TranscendentalRoot', 'isolate_real_roots', 'real_roots_of',
    'integrating_factor_xy', 'integrating_factor_p', 'is_linearizable', 'linearize', 'rectify_symmetries',
    'dsolve_second_order', 'FirstIntegral', 'Linearization',
    'abel_invariants', 'abel_equivalence', 'abel_by_invariants', 'air_solution', 'particular_solution',
    'AbelInvariants', 'AbelTransformation', 'thue', 'units_of_order', 'elements_of_norm', 'ThueEquation']
