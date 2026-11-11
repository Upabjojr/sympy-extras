//! Port of `sympy_extras.polys.numberfields`: the modules which need no
//! expressions (`minpoly` waits for the core; its two resultants are in
//! `qqbar`).

pub mod basis;
pub mod qqbar;
pub mod recognize;
