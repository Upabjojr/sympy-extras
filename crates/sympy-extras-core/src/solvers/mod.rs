//! Port of `sympy_extras.solvers`: the modules which need no expressions.

pub mod bivariate;
pub mod elimination;
pub mod exponential_diophantine;
pub mod inequalities;
pub mod integers;
pub mod isolation;
pub mod lie;
pub mod linsolve;
pub mod parametric;
pub mod rootclassification;
pub mod simplex;
#[allow(clippy::module_inception)]
pub mod solvers;
pub mod solveset;
pub mod stability;
pub mod thue_equation;
pub mod transcendental;
