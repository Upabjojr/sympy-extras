//! Port of `sympy_extras/polys/rationaltools.py`: [`together`], a sum of
//! rational expressions over a common denominator without expanding or
//! cancelling, bottom up by [`gcd_terms_sum`].

use crate::core::basic::{Basic, Head};
use crate::core::exprtools::gcd_terms_sum;
use crate::core::function::rebuild_evaluated;
use crate::core::power::pow;
use crate::error::PolyResult;

/// The sums of `e` over common denominators, bottom up, with the common
/// factors of their terms taken out; the arguments of functions too with
/// `deep`; with `fraction=False` the denominators of the terms are kept
/// apart (`together`).
pub fn together(e: &Basic, deep: bool, fraction: bool) -> PolyResult<Basic> {
    if e.args().is_empty() || (matches!(e.head(), Head::Function(_)) && !deep) {
        return Ok(e.clone());
    }
    if e.is_add() {
        let mut terms: Vec<Basic> = Vec::new();
        for a in e.args() {
            terms.push(together(a, deep, fraction)?);
        }
        return gcd_terms_sum(&terms, false, true, fraction);
    }
    if e.is_pow() {
        let base: Basic = together(&e.args()[0], deep, fraction)?;
        let exponent: Basic = if deep {
            together(&e.args()[1], deep, fraction)?
        } else {
            e.args()[1].clone()
        };
        return pow(&base, &exponent);
    }
    let mut args: Vec<Basic> = Vec::new();
    for a in e.args() {
        args.push(together(a, deep, fraction)?);
    }
    rebuild_evaluated(e, args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::add::add;
    use crate::core::printing::sstr;
    use crate::core::Symbol;
    use crate::functions::elementary::exponential::exp;

    #[test]
    fn module_examples() {
        let x: Basic = Symbol::new("x", &[]).0;
        let y: Basic = Symbol::new("y", &[]).0;
        let inv = |e: &Basic| pow(e, &Basic::negative_one()).unwrap();
        let s: Basic = add(&[inv(&x), inv(&y)]).unwrap();
        assert_eq!(sstr(&together(&s, false, true).unwrap()), "(x + y)/(x*y)");
        let s2: Basic = add(&[inv(&x), inv(&pow(&y, &Basic::int(2)).unwrap())]).unwrap();
        let _ = s2;
        let e: Basic = exp(&s).unwrap();
        assert_eq!(sstr(&together(&e, false, true).unwrap()), "exp(1/y + 1/x)");
        assert_eq!(sstr(&together(&e, true, true).unwrap()), "exp((x + y)/(x*y))");
    }
}
