//! Port of `sympy_extras/calculus/util.py` (SymPy's `periodicity` and
//! `lcim`, with the periods of the trigonometric functions and the
//! decomposition of `decompogen` written down): the period of a function
//! found structurally, sums, products and powers by the least common
//! multiple of the periods of their parts. The checks by substitution use
//! the native `simplify` of the difference; where it is not ported, the
//! check fails (as a time limit does in the Python).

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::{as_coeff_mul, as_independent, is_true};
use crate::core::symbol::Dummy;
use crate::error::{PolyError, PolyResult};
use crate::functions::elementary::complexes::{abs_, im, re};
use crate::functions::elementary::exponential::exp;
use crate::functions::elementary::trigonometric::sin;

type O = PolyResult<Option<Basic>>;

fn div(a: &Basic, b: &Basic) -> PolyResult<Basic> {
    crate::core::mul::div(a, b)
}

fn subs(e: &Basic, old: &Basic, new: &Basic) -> PolyResult<Basic> {
    crate::core::subs::subs(e, old, new)
}

fn expand_mul(e: &Basic) -> PolyResult<Basic> {
    crate::core::function::expand_mul(e, true)
}

fn function_of(e: &Basic) -> Option<Function> {
    match e.head() {
        Head::Function(f) => Some(*f),
        _ => None,
    }
}

fn general_period(f: &Basic) -> Option<Basic> {
    let pi: Basic = Basic::pi();
    match function_of(f)? {
        Function::Sin | Function::Cos | Function::Sec | Function::Csc => {
            crate::core::mul::mul(&[Basic::int(2), pi]).ok()
        }
        Function::Tan | Function::Cot => Some(pi),
        _ => None,
    }
}

fn is_trigonometric(f: &Basic) -> bool {
    general_period(f).is_some()
}

/// The period of a trigonometric function of an argument linear in
/// `symbol` (`TrigonometricFunction._period`).
fn trigonometric_period(f: &Basic, symbol: &Basic) -> O {
    let Some(general) = general_period(f) else {
        return Ok(None);
    };
    let argument: Basic = expand_mul(&f.args()[0])?;
    if !argument.has(symbol) {
        return Ok(Some(Basic::zero()));
    }
    if argument == *symbol {
        return Ok(Some(general));
    }
    if argument.is_mul() {
        let (g, h) = as_independent(&argument, symbol, false)?;
        if h == *symbol {
            return Ok(Some(div(&general, &abs_(&g)?)?));
        }
    }
    if argument.is_add() {
        let (_, h) = as_independent(&argument, symbol, true)?;
        let (g, h) = as_independent(&h, symbol, false)?;
        if h == *symbol {
            return Ok(Some(div(&general, &abs_(&g)?)?));
        }
    }
    Ok(None)
}

/// Whether two expressions are recognised equal: structurally or by the
/// native simplification of their difference (`_same`).
fn same(a: &Basic, b: &Basic) -> PolyResult<bool> {
    if a == b {
        return Ok(true);
    }
    let difference: Basic = crate::core::add::sub(a, b)?;
    let simplified = crate::timeout::attempt_within(crate::settings::settings().timeout, || {
        crate::simplify::simplify::simplify(&difference, &crate::simplify::simplify::Options::default())
    })?;
    Ok(simplified.is_some_and(|d| d == Basic::zero()))
}

fn checked(orig_f: &Basic, symbol: &Basic, period: &Basic) -> O {
    let shifted: Basic = subs(
        orig_f,
        symbol,
        &crate::core::add::add(&[symbol.clone(), period.clone()])?,
    )?;
    Ok(if same(&shifted, orig_f)? {
        Some(period.clone())
    } else {
        None
    })
}

fn generators(f: &Basic, symbol: &Basic) -> Option<Vec<Basic>> {
    fn visit(e: &Basic, symbol: &Basic, found: &mut Vec<Basic>) -> bool {
        if !e.free_symbols().contains(symbol) {
            return true;
        }
        if e.is_add() || e.is_mul() {
            return e.args().iter().all(|a| visit(a, symbol, found));
        }
        if e.is_pow()
            && crate::core::expr::is_integer_class(&e.args()[1])
            && e.args()[1]
                .as_rational()
                .is_some_and(|v| *v > crate::numbers::Rat::ZERO)
        {
            return visit(&e.args()[0], symbol, found);
        }
        if !found.contains(e) {
            found.push(e.clone());
        }
        true
    }
    let mut found: Vec<Basic> = Vec::new();
    if visit(f, symbol, &mut found) {
        Some(found)
    } else {
        None
    }
}

/// `f` as a composition, outermost first (`decompogen`; a polynomial
/// stays one piece).
fn decompogen(f: &Basic, symbol: &Basic) -> PolyResult<Vec<Basic>> {
    if !f.free_symbols().contains(symbol) {
        return Ok(vec![f.clone()]);
    }
    if matches!(f.head(), Head::Function(_)) || f.is_pow() {
        let argument: Basic = if f.is_pow() {
            f.args()[0].clone()
        } else {
            if f.args().len() != 1 {
                return Ok(vec![f.clone()]);
            }
            f.args()[0].clone()
        };
        if argument == *symbol {
            return Ok(vec![f.clone()]);
        }
        let mut result: Vec<Basic> = vec![subs(f, &argument, symbol)?];
        result.extend(decompogen(&argument, symbol)?);
        return Ok(result);
    }
    if let Some(gens) = generators(f, symbol) {
        if gens.len() == 1 && gens[0] != *symbol && gens[0] != *f {
            let g: &Basic = &gens[0];
            let outer: Basic = subs(f, g, symbol)?;
            if outer.free_symbols().contains(symbol) && !outer.has(g) {
                let mut result: Vec<Basic> = vec![outer];
                result.extend(decompogen(g, symbol)?);
                return Ok(result);
            }
        }
    }
    Ok(vec![f.clone()])
}

fn compogen(g_s: &[Basic], symbol: &Basic) -> PolyResult<Basic> {
    if g_s.len() == 1 {
        return Ok(g_s[0].clone());
    }
    let inner: Basic = compogen(&g_s[1..], symbol)?;
    subs(&g_s[0], symbol, &inner)
}

/// The period of `f` in `symbol` (`0` when `f` does not depend on it),
/// `None` when it is not periodic or no period is found; with `check`
/// the period verified by substitution (`periodicity`).
pub fn periodicity(f: &Basic, symbol: &Basic, check: bool) -> O {
    let temp: Basic = Dummy::new("x", &[(Predicate::real, true)]).0;
    let g: Basic = match f.head() {
        Head::Relational(_) => crate::core::add::sub(&f.args()[0], &f.args()[1])?,
        _ if crate::core::logic::is_boolean(f) || matches!(f.head(), Head::Tuple) => {
            return Err(PolyError::Type(format!(
                "periodicity takes an expression or a relation, not {}",
                f
            )))
        }
        _ => f.clone(),
    };
    let g: Basic = subs(&g, symbol, &temp)?;
    let Some(found) = periodicity_of(&g, &temp, check)? else {
        return Ok(None);
    };
    Ok(Some(subs(&found, &temp, symbol)?))
}

fn periodicity_of(f0: &Basic, symbol: &Basic, check: bool) -> O {
    let mut orig_f: Basic = f0.clone();
    let mut period: Option<Basic> = None;
    let mut f: Basic = crate::simplify::simplify::simplify(f0, &crate::simplify::simplify::Options::default())?;
    if !f.free_symbols().contains(symbol) {
        return Ok(Some(Basic::zero()));
    }
    if is_trigonometric(&f) {
        period = trigonometric_period(&f, symbol)?;
    }
    if f.is_function(Function::Abs) {
        let mut a: Basic = f.args()[0].clone();
        if matches!(function_of(&a), Some(Function::Sec | Function::Csc | Function::Cos)) {
            a = sin(&a.args()[0])?;
        }
        period = periodicity_of(&a, symbol, false)?;
        if let Some(p) = &period {
            if a.is_function(Function::Sin) {
                orig_f = abs_(&a)?;
                if let Some(half) = checked(&orig_f, symbol, &div(p, &Basic::int(2))?)? {
                    return Ok(Some(half));
                }
            }
        }
    }
    let is_e_power = |x: &Basic| x.is_pow() && matches!(x.args()[0].head(), Head::Exp1);
    if f.is_function(Function::Exp) || is_e_power(&f) {
        let exponent: Basic = if f.is_function(Function::Exp) {
            f.args()[0].clone()
        } else {
            f.args()[1].clone()
        };
        f = exp(&expand_mul(&exponent)?)?;
        if im(&f)? != Basic::zero() {
            let period_real = periodicity_of(&re(&f)?, symbol, false)?;
            let period_imag = periodicity_of(&im(&f)?, symbol, false)?;
            if let (Some(a), Some(b)) = (period_real, period_imag) {
                period = lcim(&[a, b])?;
            }
        }
    }
    if f.is_pow() && !is_e_power(&f) {
        let (base, expo) = (f.args()[0].clone(), f.args()[1].clone());
        let (bs, es) = (base.has(symbol), expo.has(symbol));
        period = if bs && !es {
            periodicity_of(&base, symbol, false)?
        } else if es && !bs {
            periodicity_of(&expo, symbol, false)?
        } else {
            periodicity_args(&[base, expo], symbol)?
        };
    } else if f.is_mul() {
        let (coeff, rest) = as_independent(&f, symbol, false)?;
        period = if is_trigonometric(&rest) || coeff != Basic::one() {
            periodicity_of(&rest, symbol, false)?
        } else {
            periodicity_args(rest.args(), symbol)?
        };
    } else if f.is_add() {
        let (k, rest) = as_independent(&f, symbol, true)?;
        if k != Basic::zero() {
            return periodicity_of(&rest, symbol, false);
        }
        period = periodicity_args(rest.args(), symbol)?;
    } else if f.is_function(Function::Mod) {
        let (a, n) = (f.args()[0].clone(), f.args()[1].clone());
        if a == *symbol {
            period = Some(n);
        } else if is_trigonometric(&a) {
            period = periodicity_of(&a, symbol, false)?;
        } else if !n.free_symbols().contains(symbol)
            && crate::core::expr::polynomial_like(
                &a,
                std::slice::from_ref(symbol),
                crate::core::expr::PolynomialKind::Polynomial,
            ) == Some(true)
        {
            let d: Basic = crate::core::function::derivative(&a, symbol)?;
            if !d.free_symbols().contains(symbol) && d != Basic::zero() {
                period = Some(abs_(&div(&n, &d)?)?);
            }
        }
    } else if matches!(f.head(), Head::Piecewise) {
        // not handled, as in SymPy
    } else if period.is_none() {
        let g_s: Vec<Basic> = decompogen(&f, symbol)?;
        let count: usize = g_s.len();
        if count > 1 {
            for index in 0..count {
                let start: usize = count - 1 - index;
                let g: Basic = compogen(&g_s[start..], symbol)?;
                if g != orig_f && g != f {
                    period = periodicity_of(&g, symbol, false)?;
                    if period.is_some() {
                        break;
                    }
                }
            }
        }
    }
    match period {
        Some(p) if check => checked(&orig_f, symbol, &p),
        other => Ok(other),
    }
}

fn periodicity_args(args: &[Basic], symbol: &Basic) -> O {
    let mut periods: Vec<Basic> = Vec::new();
    for f in args {
        let Some(period) = periodicity_of(f, symbol, false)? else {
            return Ok(None);
        };
        if period != Basic::zero() {
            periods.push(period);
        }
    }
    if periods.len() > 1 {
        return lcim(&periods);
    }
    Ok(periods.into_iter().next())
}

/// The least common multiple of rational numbers, or of irrational
/// numbers which are rational multiples of one term (`lcim`).
pub fn lcim(numbers: &[Basic]) -> O {
    if numbers.iter().all(|n| is_true(n, Predicate::irrational)) {
        let factors: Vec<(Basic, Basic)> = numbers
            .iter()
            .map(|n| Ok(as_coeff_mul(&crate::polys::polytools::factor(n, &[], false, true)?)))
            .collect::<PolyResult<_>>()?;
        let term: Basic = factors[0].1.clone();
        if factors.iter().all(|(_, rest)| *rest == term) {
            let coefficients: Vec<Basic> = factors.into_iter().map(|(c, _)| c).collect();
            return Ok(Some(crate::core::mul::mul(&[
                crate::polys::polytools::lcm_list(&coefficients, &[])?,
                term,
            ])?));
        }
        return Ok(None);
    }
    if numbers.iter().all(|n| is_true(n, Predicate::rational)) {
        return Ok(Some(crate::polys::polytools::lcm_list(numbers, &[])?));
    }
    Ok(None)
}
