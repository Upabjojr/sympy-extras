//! Port of `sympy_extras.solvers`: the modules which need no expressions.

pub mod bivariate;
pub mod exponential_diophantine;
pub mod integers;
pub mod linsolve;
pub mod parametric;
pub mod simplex;
#[allow(clippy::module_inception)]
pub mod solvers;
pub mod thue_equation;
