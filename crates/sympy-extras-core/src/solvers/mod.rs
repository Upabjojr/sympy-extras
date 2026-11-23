//! Port of `sympy_extras.solvers`: the modules which need no expressions.

pub mod abel;
pub mod bivariate;
pub mod charpit;
pub mod dae;
pub mod darboux;
pub mod dsolve;
pub mod elimination;
pub mod exponential_diophantine;
pub mod factorization;
pub mod first_order;
pub mod inequalities;
pub mod integers;
pub mod irregular;
pub mod isolation;
pub mod kovacic;
pub mod lie;
pub mod linear_ode;
pub mod linear_systems;
pub mod linsolve;
pub mod ode;
pub mod ode_cases;
pub mod parametric;
pub mod pde;
pub mod recurr;
pub mod rootclassification;
pub mod second_order;
pub mod simplex;
#[allow(clippy::module_inception)]
pub mod solvers;
pub mod solveset;
pub mod special;
pub mod stability;
pub mod thue_equation;
pub mod transcendental;
