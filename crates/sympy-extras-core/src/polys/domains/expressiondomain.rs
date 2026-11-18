//! Port of `sympy_extras/polys/domains/expressiondomain.py` (itself
//! ported from SymPy's `ExpressionDomain`): the field `EX` of the
//! expressions of the core, whose results are put in the normal form
//! `cancel(e)` expanded over products only, and `EXR`, which also
//! rationalizes the denominators with square roots of numbers by
//! `radsimp`. The gcd of two elements is `1`, as in SymPy.
//!
//! The arithmetic of the [`Domain`] trait cannot fail; where the core
//! arithmetic or the normal form fails (a case not ported), the result is
//! the unevaluated sum or product.

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is_true;
use crate::error::{zero_division, PolyResult};
use crate::numbers::*;

use super::domain::Domain;

/// The domain of expressions (`EX`), or with the denominators
/// rationalized (`EXR`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpressionDomain {
    /// `RationalizedExpressionDomain`: `radsimp` after the normal form
    /// of an expression with square roots of numbers.
    pub rationalized: bool,
}

/// The domain of expressions.
pub const EX: ExpressionDomain = ExpressionDomain { rationalized: false };
/// The domain of expressions, the denominators rationalized.
pub const EXR: ExpressionDomain = ExpressionDomain { rationalized: true };

/// SymPy's `Expression.simplify`: `cancel(e)` expanded over the products
/// (`ex_normal`).
pub fn ex_normal(e: &Basic) -> PolyResult<Basic> {
    let hints = crate::core::function::Hints {
        mul: true,
        multinomial: false,
        power_base: false,
        power_exp: false,
        log: false,
        force: false,
        complex: false,
        trig: false,
        func: false,
    };
    crate::core::function::expand_with(&crate::polys::polytools::cancel(e, &[])?, false, hints)
}

fn has_numeric_sqrt(e: &Basic) -> bool {
    e.preorder().iter().any(|node| {
        node.is_pow()
            && node.args()[1].is_rational_class()
            && node.args()[1].as_rational().is_some_and(|v| denom(v) == Int::from(2))
            && node.args()[0].free_symbols().is_empty()
    })
}

impl ExpressionDomain {
    /// The normal form of the result of an operation.
    pub fn normal(&self, e: &Basic) -> PolyResult<Basic> {
        let value: Basic = ex_normal(e)?;
        if self.rationalized && has_numeric_sqrt(&value) {
            return ex_normal(&crate::simplify::radsimp::radsimp(&value, true, 4)?);
        }
        Ok(value)
    }

    fn normal_or(&self, e: PolyResult<Basic>, raw: impl FnOnce() -> Basic) -> Basic {
        match e.and_then(|v| self.normal(&v)) {
            Ok(v) => v,
            Err(_) => raw(),
        }
    }

    /// The numerator (`numer`).
    pub fn numer(&self, a: &Basic) -> PolyResult<Basic> {
        Ok(crate::core::expr::as_numer_denom(a)?.0)
    }

    /// The denominator (`denom`).
    pub fn denom(&self, a: &Basic) -> PolyResult<Basic> {
        Ok(crate::core::expr::as_numer_denom(a)?.1)
    }
}

fn minus(a: &Basic) -> Basic {
    crate::core::mul::neg(a).unwrap_or_else(|_| crate::core::mul::raw_mul(vec![Basic::negative_one(), a.clone()]))
}

impl Domain for ExpressionDomain {
    type Elem = Basic;

    fn to_expr(&self, a: &Basic) -> PolyResult<Basic> {
        Ok(a.clone())
    }

    fn from_expr(&self, e: &Basic) -> PolyResult<Basic> {
        Ok(e.clone())
    }

    fn name(&self) -> String {
        "EX".to_string()
    }

    fn is_field(&self) -> bool {
        true
    }

    fn is_ordered(&self) -> bool {
        false
    }

    fn characteristic(&self) -> Int {
        Int::ZERO
    }

    fn zero(&self) -> Basic {
        Basic::zero()
    }

    fn one(&self) -> Basic {
        Basic::one()
    }

    fn is_zero(&self, a: &Basic) -> bool {
        if a.is_number_class() {
            return *a == Basic::zero();
        }
        is_true(a, Predicate::zero)
    }

    fn is_one(&self, a: &Basic) -> bool {
        *a == Basic::one()
    }

    /// SymPy's `EX.is_negative`: `could_extract_minus_sign`.
    fn is_negative(&self, a: &Basic) -> bool {
        crate::core::printing::could_extract_minus_sign(a)
    }

    fn is_unit(&self, a: &Basic) -> bool {
        !self.is_zero(a)
    }

    fn add(&self, a: &Basic, b: &Basic) -> Basic {
        if self.is_zero(b) {
            return a.clone();
        }
        if self.is_zero(a) {
            return b.clone();
        }
        self.normal_or(crate::core::add::add(&[a.clone(), b.clone()]), || {
            crate::core::add::raw_add(vec![a.clone(), b.clone()])
        })
    }

    fn sub(&self, a: &Basic, b: &Basic) -> Basic {
        if self.is_zero(b) {
            return a.clone();
        }
        if self.is_zero(a) {
            return self.neg(b);
        }
        let mb: Basic = minus(b);
        self.normal_or(crate::core::add::add(&[a.clone(), mb.clone()]), || {
            crate::core::add::raw_add(vec![a.clone(), mb])
        })
    }

    fn neg(&self, a: &Basic) -> Basic {
        minus(a)
    }

    fn mul(&self, a: &Basic, b: &Basic) -> Basic {
        if self.is_zero(a) || self.is_zero(b) {
            return Basic::zero();
        }
        let raw = || crate::core::mul::raw_mul(vec![a.clone(), b.clone()]);
        if a.is_number_class() && b.is_number_class() {
            return crate::core::mul::mul(&[a.clone(), b.clone()]).unwrap_or_else(|_| raw());
        }
        self.normal_or(crate::core::mul::mul(&[a.clone(), b.clone()]), raw)
    }

    fn pow(&self, a: &Basic, n: u64) -> Basic {
        let exponent: Basic = Basic::integer(Int::from(n));
        self.normal_or(crate::core::power::pow(a, &exponent), || {
            Basic::new(Head::Pow, vec![a.clone(), exponent.clone()])
        })
    }

    fn exquo(&self, a: &Basic, b: &Basic) -> PolyResult<Basic> {
        if self.is_zero(b) {
            return Err(zero_division("division by zero in EX"));
        }
        let inverse: Basic = crate::core::power::pow(b, &Basic::negative_one())?;
        self.normal(&crate::core::mul::mul(&[a.clone(), inverse])?)
    }

    fn quo(&self, a: &Basic, b: &Basic) -> PolyResult<Basic> {
        self.exquo(a, b)
    }

    fn rem(&self, _a: &Basic, b: &Basic) -> PolyResult<Basic> {
        if self.is_zero(b) {
            return Err(zero_division("division by zero in EX"));
        }
        Ok(Basic::zero())
    }

    fn invert(&self, a: &Basic) -> PolyResult<Basic> {
        if self.is_zero(a) {
            return Err(zero_division("0 has no inverse"));
        }
        self.normal(&crate::core::power::pow(a, &Basic::negative_one())?)
    }

    /// `1`, as SymPy's `EX.gcd`.
    fn gcd(&self, _a: &Basic, _b: &Basic) -> Basic {
        Basic::one()
    }

    /// SymPy's `EX.lcm`: the `lcm` of the expressions.
    fn lcm(&self, a: &Basic, b: &Basic) -> Basic {
        crate::polys::polytools::lcm_expr(a, b, &[])
            .unwrap_or_else(|_| crate::core::mul::raw_mul(vec![a.clone(), b.clone()]))
    }

    fn gcdex(&self, a: &Basic, b: &Basic) -> PolyResult<(Basic, Basic, Basic)> {
        if !self.is_zero(a) {
            return Ok((self.invert(a)?, Basic::zero(), Basic::one()));
        }
        if !self.is_zero(b) {
            return Ok((Basic::zero(), self.invert(b)?, Basic::one()));
        }
        Ok((Basic::zero(), Basic::zero(), Basic::zero()))
    }

    fn from_int(&self, n: i64) -> Basic {
        Basic::int(n)
    }

    fn from_integer(&self, n: &Int) -> Basic {
        Basic::integer(n.clone())
    }

    fn canonical_unit(&self, a: &Basic) -> Basic {
        if self.is_zero(a) {
            return Basic::one();
        }
        self.invert(a).unwrap_or_else(|_| Basic::one())
    }
}
