//! Port of `sympy_extras/polys/polyutils.py`: the conversions between the
//! expressions of the core and the terms of the polynomials, without
//! expanding the expression ([`dict_from_expr`] multiplies out with the
//! ring arithmetic; a part free of the generators is a coefficient,
//! converted by [`Domain::from_expr`]) and back ([`expr_from_dict`], with
//! the evaluating `add`, `mul`, `pow`).

use std::collections::HashMap;

use crate::core::add::add;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is_integer_class;
use crate::core::mul::mul;
use crate::core::power::pow;
use crate::core::Symbol;
use crate::error::{PolyError, PolyResult};
use crate::numbers::*;

use super::domains::domain::Domain;
use super::rings::{Monomial, Terms};

fn add_terms<K: Domain>(a: Terms<K::Elem>, b: Terms<K::Elem>, k: &K) -> Terms<K::Elem> {
    let mut result: Terms<K::Elem> = a;
    for (m, c) in b {
        match result.get(&m) {
            Some(existing) => {
                let s: K::Elem = k.add(existing, &c);
                if k.is_zero(&s) {
                    result.remove(&m);
                } else {
                    result.insert(m, s);
                }
            }
            None => {
                if !k.is_zero(&c) {
                    result.insert(m, c);
                }
            }
        }
    }
    result
}

fn mul_terms<K: Domain>(a: &Terms<K::Elem>, b: &Terms<K::Elem>, k: &K) -> Terms<K::Elem> {
    let mut result: Terms<K::Elem> = Terms::new();
    for (m1, c1) in a {
        for (m2, c2) in b {
            let m: Monomial = m1.iter().zip(m2).map(|(i, j)| i + j).collect();
            let product: K::Elem = k.mul(c1, c2);
            match result.get(&m) {
                Some(existing) => {
                    let s: K::Elem = k.add(existing, &product);
                    if k.is_zero(&s) {
                        result.remove(&m);
                    } else {
                        result.insert(m, s);
                    }
                }
                None => {
                    if !k.is_zero(&product) {
                        result.insert(m, product);
                    }
                }
            }
        }
    }
    result
}

fn pow_terms<K: Domain>(a: &Terms<K::Elem>, mut n: u64, one: &Terms<K::Elem>, k: &K) -> Terms<K::Elem> {
    let mut result: Terms<K::Elem> = one.clone();
    let mut base: Terms<K::Elem> = a.clone();
    while n > 0 {
        if n & 1 == 1 {
            result = mul_terms(&result, &base, k);
        }
        n >>= 1;
        if n > 0 {
            base = mul_terms(&base, &base, k);
        }
    }
    result
}

/// The name of a symbol (a `Symbol`, `Dummy` or `Wild`: the Python
/// subclasses of `Symbol`).
fn symbol_name(e: &Basic) -> Option<&str> {
    match e.head() {
        Head::Symbol(s) | Head::Dummy(s, _) => Some(&s.name),
        Head::Wild(w) => Some(&w.symbol.name),
        _ => None,
    }
}

/// The terms `{exponents: coefficient}` of a core expression which is a
/// polynomial in the generators named `gens` with coefficients in `k`
/// (`dict_from_expr`); `Value` when it is not one.
pub fn dict_from_expr<K: Domain>(expr: &Basic, gens: &[String], k: &K) -> PolyResult<Terms<K::Elem>> {
    let index: HashMap<&str, usize> = gens.iter().enumerate().map(|(i, g)| (g.as_str(), i)).collect();
    let zero_monomial: Monomial = vec![0; gens.len()];
    let mut one: Terms<K::Elem> = Terms::new();
    one.insert(zero_monomial.clone(), k.one());
    walk(expr, gens, &index, &zero_monomial, &one, k)
}

fn walk<K: Domain>(
    e: &Basic,
    gens: &[String],
    index: &HashMap<&str, usize>,
    zero_monomial: &Monomial,
    one: &Terms<K::Elem>,
    k: &K,
) -> PolyResult<Terms<K::Elem>> {
    if let Some(&i) = symbol_name(e).and_then(|name| index.get(name)) {
        let mut m: Monomial = zero_monomial.clone();
        m[i] = 1;
        let mut t: Terms<K::Elem> = Terms::new();
        t.insert(m, k.one());
        return Ok(t);
    }
    let depends: bool = e
        .free_symbols()
        .iter()
        .any(|s| symbol_name(s).is_some_and(|name| index.contains_key(name)));
    if !depends {
        let c: K::Elem = k.from_expr(e)?;
        let mut t: Terms<K::Elem> = Terms::new();
        if !k.is_zero(&c) {
            t.insert(zero_monomial.clone(), c);
        }
        return Ok(t);
    }
    match e.head() {
        Head::Add => {
            let mut total: Terms<K::Elem> = Terms::new();
            for a in e.args() {
                total = add_terms(total, walk(a, gens, index, zero_monomial, one, k)?, k);
            }
            return Ok(total);
        }
        Head::Mul => {
            let mut product: Terms<K::Elem> = one.clone();
            for a in e.args() {
                product = mul_terms(&product, &walk(a, gens, index, zero_monomial, one, k)?, k);
            }
            return Ok(product);
        }
        Head::Pow => {
            let exponent: &Basic = &e.args()[1];
            if is_integer_class(exponent) {
                let n: &Int = exponent.as_rational().expect("an integer").numerator();
                if !is_negative(n) {
                    let n: u64 = u64::try_from(n).map_err(|_| PolyError::Value("a huge exponent".to_string()))?;
                    let base: Terms<K::Elem> = walk(&e.args()[0], gens, index, zero_monomial, one, k)?;
                    return Ok(pow_terms(&base, n, one, k));
                }
            }
        }
        _ => {}
    }
    Err(PolyError::Value(format!(
        "{} is not a polynomial in {}",
        e,
        gens.join(", ")
    )))
}

/// The core expression of the terms, the generators named `gens`
/// (`expr_from_dict`).
pub fn expr_from_dict<K: Domain>(terms: &Terms<K::Elem>, gens: &[String], k: &K) -> PolyResult<Basic> {
    let symbols: Vec<Basic> = gens.iter().map(|name| Symbol::new(name, &[]).0).collect();
    let mut parts: Vec<Basic> = Vec::new();
    for (m, c) in terms {
        let mut factors: Vec<Basic> = vec![k.to_expr(c)?];
        for (s, e) in symbols.iter().zip(m) {
            if *e > 1 {
                factors.push(pow(s, &Basic::int(i64::from(*e)))?);
            } else if *e == 1 {
                factors.push(s.clone());
            }
        }
        parts.push(mul(&factors)?);
    }
    if parts.is_empty() {
        return Ok(Basic::zero());
    }
    add(&parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::printing::sstr;
    use crate::polys::domains::rationalfield::QQ;

    #[test]
    fn module_example() {
        let x: Basic = Symbol::new("x", &[]).0;
        let y: Basic = Symbol::new("y", &[]).0;
        let square: Basic = pow(&add(&[x.clone(), y]).unwrap(), &Basic::int(2)).unwrap();
        let e: Basic = add(&[square, mul(&[Basic::from_rational(rat(-1, 2)), x]).unwrap()]).unwrap();
        let gens: Vec<String> = vec!["x".to_string(), "y".to_string()];
        let terms = dict_from_expr(&e, &gens, &QQ).unwrap();
        assert_eq!(terms.len(), 4);
        assert_eq!(terms[&vec![1, 0]], rat(-1, 2));
        assert_eq!(terms[&vec![1, 1]], rat(2, 1));
        assert_eq!(
            sstr(&expr_from_dict(&terms, &gens, &QQ).unwrap()),
            "x**2 + 2*x*y - x/2 + y**2"
        );
    }
}
