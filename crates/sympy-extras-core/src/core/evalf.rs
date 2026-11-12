//! Port of the numerical evaluation of `sympy_extras/core/evalf.py` used by
//! the assumptions: `numeric_value`, the value of a numerical expression as
//! a complex number at a working precision (bits), Python's `mpmath.mpc`
//! under `workprec`.
//!
//! The numbers are `dashu-cmplx` complex numbers of `dashu-float` binary
//! floats. As in `mpmath`, the results are computed at the working
//! precision with a few guard bits and are not correctly rounded; the
//! callers compare them with thresholds. `z**w` is `exp(w ln z)` (integer
//! exponents by repeated squaring): `dashu-cmplx` 0.6.2's own `powf` does not
//! return when the exact result has a zero part (`(-1)**(1/2)`).
//!
//! The constants which `dashu-float` does not have are computed here: Euler's
//! `EulerGamma` by Brent and McMillan's algorithm, `Catalan` by Ramanujan's
//! series, `GoldenRatio` and `TribonacciConstant` by their radicals.
//!
//! References
//! ==========
//!
//! 1. R. P. Brent, E. M. McMillan, *Some new algorithms for high-precision
//!    computation of Euler's constant*, Mathematics of Computation 34 (1980),
//!    305-312 (algorithm B1).
//! 2. B. C. Berndt, *Ramanujan's Notebooks*, Part I, Springer, 1985, p. 289
//!    (the series of Catalan's constant).

use dashu_cmplx::CBig;
use dashu_float::round::mode::HalfEven;
use dashu_float::FBig;

use crate::numbers::*;

use super::basic::{Basic, Head};

/// A binary float.
pub type Real = FBig<HalfEven, 2>;

/// A complex number of two binary floats.
pub type Complex = CBig<HalfEven, 2>;

/// Guard bits of the intermediate computations.
const GUARD: usize = 20;

/// An integer as a float of the given precision.
pub fn real_int(n: &Int, prec: usize) -> Real {
    Real::from(n.clone()).with_precision(prec).value()
}

/// A rational as a float of the given precision.
pub fn real_rat(r: &Rat, prec: usize) -> Real {
    real_int(r.numerator(), prec + GUARD) / real_int(&denom(r), prec + GUARD)
}

pub fn complex(re: Real, im: Real) -> Complex {
    Complex::from_parts(re, im)
}

fn real_zero(prec: usize) -> Real {
    Real::ZERO.with_precision(prec).value()
}

pub fn complex_real(re: Real, prec: usize) -> Complex {
    complex(re, real_zero(prec))
}

/// Euler's constant by Brent and McMillan's algorithm B1: with `n` about
/// `prec ln 2 / 4`, `gamma = U/V - O(e^(-4n))` for the sums of `A_k` and `B_k`.
pub fn euler_gamma(prec: usize) -> Real {
    let wp: usize = prec + GUARD + 10;
    let n: u64 = (prec as u64 * 7) / 40 + 2; // prec ln 2 / 4, with margin
    let nn: Real = real_int(&Int::from(n * n), wp);
    let ln_n: Real = real_int(&Int::from(n), wp).ln();
    let mut a: Real = -ln_n;
    let mut b: Real = real_int(&Int::ONE, wp);
    let mut u: Real = a.clone();
    let mut v: Real = b.clone();
    let steps: u64 = 4 * n + 10; // the terms decay past k = alpha n with alpha = 3.59...
    for k in 1..=steps {
        let kk: Real = real_int(&Int::from(k), wp);
        b = &b * &nn / (&kk * &kk);
        a = (&a * &nn / &kk + &b) / &kk;
        u = &u + &a;
        v = &v + &b;
    }
    (u / v).with_precision(prec).value()
}

/// Catalan's constant by Ramanujan's series `G = pi/8 ln(2 + sqrt(3)) + 3/8
/// sum_{k >= 0} 1/((2k + 1)**2 binomial(2k, k))`.
pub fn catalan(prec: usize) -> Real {
    let wp: usize = prec + GUARD + 10;
    let one: Real = real_int(&Int::ONE, wp);
    let mut total: Real = real_zero(wp);
    let mut binomial: Int = Int::ONE; // binomial(2k, k)
    let mut k: u64 = 0;
    loop {
        let odd: Int = Int::from(2 * k + 1);
        let denominator: Int = &odd * &odd * &binomial;
        // the terms below 2**(-wp) are negligible
        if bits(&denominator) > wp as u64 + 2 {
            break;
        }
        let term: Real = &one / real_int(&denominator, wp);
        total = &total + &term;
        k += 1;
        // binomial(2k, k) = binomial(2k - 2, k - 1) (2k)(2k - 1)/k**2
        binomial = binomial * Int::from((2 * k) * (2 * k - 1)) / Int::from(k * k);
    }
    let three: Real = real_int(&Int::from(3u8), wp);
    let eight: Real = real_int(&Int::from(8u8), wp);
    let log_term: Real = (real_int(&Int::from(2u8), wp) + three.clone().sqrt()).ln();
    (Real::pi(wp) / &eight * log_term + &three / &eight * total)
        .with_precision(prec)
        .value()
}

/// `(1 + sqrt(5))/2`.
pub fn golden_ratio(prec: usize) -> Real {
    let wp: usize = prec + GUARD;
    ((real_int(&Int::ONE, wp) + real_int(&Int::from(5u8), wp).sqrt()) / real_int(&Int::from(2u8), wp))
        .with_precision(prec)
        .value()
}

/// `(1 + cbrt(19 + 3 sqrt(33)) + cbrt(19 - 3 sqrt(33)))/3`.
pub fn tribonacci_constant(prec: usize) -> Real {
    let wp: usize = prec + GUARD;
    let s: Real = real_int(&Int::from(33u8), wp).sqrt();
    let three: Real = real_int(&Int::from(3u8), wp);
    let nineteen: Real = real_int(&Int::from(19u8), wp);
    let a: Real = (&nineteen + &three * &s).nth_root(3);
    let b: Real = (&nineteen - &three * &s).nth_root(3);
    ((real_int(&Int::ONE, wp) + a + b) / three).with_precision(prec).value()
}

fn is_zero(z: &Complex) -> bool {
    z.is_zero()
}

/// The real and the imaginary part.
pub fn parts(z: &Complex) -> (Real, Real) {
    z.clone().into_parts()
}

/// The principal square root from the parts (mpmath's `mpc_sqrt`: exact
/// on the real axis).
pub fn complex_sqrt(z: &Complex, prec: usize) -> Complex {
    let (a, b) = parts(z);
    let zero: Real = real_zero(prec);
    if b.repr().significand().is_zero() {
        if a >= Real::ZERO {
            return complex(a.sqrt(), zero);
        }
        return complex(zero, (-a).sqrt());
    }
    let two: Real = real_int(&Int::from(2u8), prec);
    let modulus: Real = (&a * &a + &b * &b).sqrt();
    // the larger half from the modulus, the other as b/(2 x): no
    // cancellation of |z| - |a| for a small imaginary part (mpmath's way)
    if a >= Real::ZERO {
        let x: Real = ((&modulus + &a) / &two).sqrt();
        let y: Real = &b / &(&two * &x);
        return complex(x, y);
    }
    let t: Real = ((&modulus - &a) / &two).sqrt();
    let x: Real = abs_of(&b) / (&two * &t);
    complex(x, if b < Real::ZERO { -t } else { t })
}

fn abs_of(r: &Real) -> Real {
    if *r < Real::ZERO {
        -r.clone()
    } else {
        r.clone()
    }
}

/// `z**w` on the principal branch: repeated squaring for an integer `w`,
/// the square root to an integer power for a half-integer (mpmath's
/// `mpc_pow_mpf`), `exp(w ln z)` otherwise (`z` nonzero).
fn power(z: &Complex, w: &Basic, wz: &Complex, prec: usize) -> Option<Complex> {
    if let Some(r) = w.as_rational() {
        if denom(r) == Int::from(2u8) && bits(&numer(r)) <= 62 {
            let n: Int = numer(r);
            let root: Complex = complex_sqrt(z, prec);
            if is_negative(&n) {
                let one: Complex = complex_real(real_int(&Int::ONE, prec), prec);
                return Some(&one / &root.powi(-n));
            }
            return Some(root.powi(n));
        }
        if r.is_int() {
            let n: Int = numer(r);
            if bits(&n) > 62 {
                // beyond the exponents of dashu's floats unless |z| = 1 exactly
                let (re, im) = parts(z);
                let unit: bool = im.repr().significand().is_zero()
                    && (re == real_int(&Int::ONE, prec) || re == real_int(&-Int::ONE, prec));
                if !unit {
                    return None;
                }
                let odd: bool = bit(&n, 0);
                let sign: Int = if odd && re < Real::ZERO { -Int::ONE } else { Int::ONE };
                return Some(complex_real(real_int(&sign, prec + GUARD), prec + GUARD));
            }
            if n.is_zero() {
                // dashu's powi(0) is a one of unlimited precision, on which
                // later operations panic
                return Some(complex_real(real_int(&Int::ONE, prec + GUARD), prec + GUARD));
            }
            if is_negative(&n) {
                let one: Complex = complex_real(real_int(&Int::ONE, prec + GUARD), prec + GUARD);
                return Some(&one / &z.powi(-n));
            }
            return Some(z.powi(n));
        }
    }
    Some((wz * &z.ln()).exp())
}

/// The value of a numerical expression at the working precision `prec`
/// (bits), `None` when `e` contains a symbol or something not evaluable (an
/// infinity, `nan`; the function classes until they are ported).
pub fn numeric_value(e: &Basic, prec: usize) -> Option<Complex> {
    let wp: usize = prec + GUARD;
    let real = |x: Real| -> Option<Complex> { Some(complex_real(x, wp)) };
    match e.head() {
        Head::Rational { value, .. } => real(real_rat(value, wp)),
        Head::IndexedRoot => crate::polys::cad::indexedroot::indexed_root_value(e, wp),
        Head::Pi => real(Real::pi(wp)),
        Head::Exp1 => real(Real::e(wp)),
        Head::EulerGamma => real(euler_gamma(wp)),
        Head::Catalan => real(catalan(wp)),
        Head::GoldenRatio => real(golden_ratio(wp)),
        Head::TribonacciConstant => real(tribonacci_constant(wp)),
        Head::ImaginaryUnit => Some(complex(real_zero(wp), real_int(&Int::ONE, wp))),
        Head::Add | Head::Mul => {
            let mut result: Complex = complex_real(real_int(&Int::from(u8::from(e.is_mul())), wp), wp);
            for a in e.args() {
                let value: Complex = numeric_value(a, prec)?;
                result = if e.is_add() { &result + &value } else { &result * &value };
            }
            Some(result)
        }
        Head::Pow => {
            let (base, exponent) = (&e.args()[0], &e.args()[1]);
            let b: Complex = numeric_value(base, prec)?;
            let x: Complex = numeric_value(exponent, prec)?;
            if is_zero(&b) {
                if is_zero(&x) {
                    return real(real_int(&Int::ONE, wp));
                }
                if parts(&x).0 > real_zero(wp) {
                    return real(real_zero(wp));
                }
                return None;
            }
            let value: Complex = power(&b, exponent, &x, wp)?;
            // mpmath's exponents are unbounded, dashu's are machine words: a
            // value beyond them has no representation here (no panic)
            let (re, im) = parts(&value);
            if re.repr().is_infinite() || im.repr().is_infinite() {
                return None;
            }
            Some(value)
        }
        Head::Function(g @ (super::basic::Function::Arg | super::basic::Function::Atan2)) => {
            let mut values: Vec<Complex> = Vec::new();
            for a in e.args() {
                values.push(numeric_value(a, prec)?);
            }
            crate::functions::elementary::argument::argument_value(*g, &values, wp)
        }
        Head::Function(g @ (super::basic::Function::DiracDelta | super::basic::Function::Heaviside)) => {
            let mut values: Vec<Complex> = Vec::new();
            for a in e.args() {
                values.push(numeric_value(a, prec)?);
            }
            crate::functions::special::delta_functions::delta_value(*g, &values)
        }
        Head::Function(g @ (super::basic::Function::Max | super::basic::Function::Min)) => {
            let mut values: Vec<Complex> = Vec::new();
            for a in e.args() {
                values.push(numeric_value(a, prec)?);
            }
            crate::functions::elementary::miscellaneous::minmax_value(*g, &values)
        }
        Head::Function(f) => {
            let v: Complex = numeric_value(&e.args()[0], prec)?;
            let value: Complex = match f {
                super::basic::Function::Exp => {
                    // beyond the exponents of dashu's floats (mpmath's are unbounded)
                    if parts(&v).0.repr().exponent() > 40 {
                        return None;
                    }
                    v.exp()
                }
                super::basic::Function::Log => {
                    if is_zero(&v) {
                        return None;
                    }
                    v.ln()
                }
                super::basic::Function::Sin
                | super::basic::Function::Cos
                | super::basic::Function::Tan
                | super::basic::Function::Cot
                | super::basic::Function::Sec
                | super::basic::Function::Csc => {
                    use super::basic::Function as F;
                    if parts(&v).1.repr().exponent() > 40 {
                        return None;
                    }
                    // sin z = (e^(iz) - e^(-iz))/(2i), cos z = (e^(iz) + e^(-iz))/2
                    let i: Complex = complex(real_zero(wp), real_int(&Int::ONE, wp));
                    let iz: Complex = &i * &v;
                    let (a, b) = (iz.exp(), (-&iz).exp());
                    let two: Complex = complex_real(real_int(&Int::from(2u8), wp), wp);
                    let s: Complex = &(&a - &b) / &(&two * &i);
                    let c: Complex = &(&a + &b) / &two;
                    match f {
                        F::Sin => s,
                        F::Cos => c,
                        F::Tan | F::Sec => {
                            if is_zero(&c) {
                                return None;
                            }
                            if matches!(f, F::Tan) {
                                &s / &c
                            } else {
                                &complex_real(real_int(&Int::ONE, wp), wp) / &c
                            }
                        }
                        _ => {
                            if is_zero(&s) {
                                return None;
                            }
                            if matches!(f, F::Cot) {
                                &c / &s
                            } else {
                                &complex_real(real_int(&Int::ONE, wp), wp) / &s
                            }
                        }
                    }
                }
                super::basic::Function::Sinh
                | super::basic::Function::Cosh
                | super::basic::Function::Tanh
                | super::basic::Function::Coth
                | super::basic::Function::Sech
                | super::basic::Function::Csch => {
                    use super::basic::Function as F;
                    if parts(&v).0.repr().exponent() > 40 {
                        return None;
                    }
                    let (a, b) = (v.exp(), (-&v).exp());
                    let two: Complex = complex_real(real_int(&Int::from(2u8), wp), wp);
                    let one: Complex = complex_real(real_int(&Int::ONE, wp), wp);
                    let s: Complex = &(&a - &b) / &two;
                    let c: Complex = &(&a + &b) / &two;
                    match f {
                        F::Sinh => s,
                        F::Cosh => c,
                        F::Tanh | F::Sech => {
                            if is_zero(&c) {
                                return None;
                            }
                            if matches!(f, F::Tanh) {
                                &s / &c
                            } else {
                                &one / &c
                            }
                        }
                        _ => {
                            if is_zero(&s) {
                                return None;
                            }
                            if matches!(f, F::Coth) {
                                &c / &s
                            } else {
                                &one / &s
                            }
                        }
                    }
                }
                g @ (super::basic::Function::Asin
                | super::basic::Function::Acos
                | super::basic::Function::Atan
                | super::basic::Function::Acot
                | super::basic::Function::Asec
                | super::basic::Function::Acsc
                | super::basic::Function::Asinh
                | super::basic::Function::Acosh
                | super::basic::Function::Atanh
                | super::basic::Function::Acoth
                | super::basic::Function::Asech
                | super::basic::Function::Acsch) => crate::functions::elementary::inverse::inverse_value(*g, &v, wp)?,
                super::basic::Function::Arg
                | super::basic::Function::Atan2
                | super::basic::Function::Max
                | super::basic::Function::Min
                | super::basic::Function::DiracDelta
                | super::basic::Function::Heaviside => {
                    unreachable!("evaluated above with all their arguments")
                }
                g
                @ (super::basic::Function::Floor | super::basic::Function::Ceiling | super::basic::Function::Frac) => {
                    crate::functions::elementary::integers::rounding_value(*g, &v)
                }
                super::basic::Function::Re => complex_real(parts(&v).0, wp),
                super::basic::Function::Im => complex_real(parts(&v).1, wp),
                super::basic::Function::Conjugate => {
                    let (re, im) = parts(&v);
                    complex(re, -im)
                }
                super::basic::Function::Abs | super::basic::Function::Sign => {
                    let (re, im) = parts(&v);
                    let modulus: Real = (&re * &re + &im * &im).sqrt();
                    if matches!(f, super::basic::Function::Abs) {
                        complex_real(modulus, wp)
                    } else if is_zero(&v) {
                        complex_real(real_zero(wp), wp)
                    } else {
                        complex(re / &modulus, im / &modulus)
                    }
                }
            };
            let (re, im) = parts(&value);
            if re.repr().is_infinite() || im.repr().is_infinite() {
                return None;
            }
            Some(value)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value as a decimal string with `digits` digits after the point.
    fn decimal(x: &Real, digits: usize) -> String {
        let scaled: Real = x * real_int(&Int::from(10u8).pow(digits), 400);
        let n: Int = scaled.round().to_int().value();
        let text: String = abs(&n).to_string();
        let (whole, fraction) = text.split_at(text.len() - digits);
        format!(
            "{}{}.{}",
            if is_negative(&n) { "-" } else { "" },
            if whole.is_empty() { "0" } else { whole },
            fraction
        )
    }

    #[test]
    fn constants_against_mpmath() {
        // mpmath at 160 bits (48 digits)
        assert_eq!(
            decimal(&euler_gamma(200), 40),
            "0.5772156649015328606065120900824024310422"
        );
        assert_eq!(decimal(&catalan(200), 40), "0.9159655941772190150546035149323841107741");
        assert_eq!(
            decimal(&golden_ratio(200), 40),
            "1.6180339887498948482045868343656381177203"
        );
        assert_eq!(
            decimal(&tribonacci_constant(200), 40),
            "1.8392867552141611325518525646532866004242"
        );
        assert_eq!(
            decimal(&Real::pi(200), 40),
            "3.1415926535897932384626433832795028841972"
        );
    }

    #[test]
    fn values_of_expressions() {
        // (-1)**(1/2) = I, 2**(1/2) - 1, (1 + I)**2 = 2 I
        let minus_one_half = Basic::new(Head::Pow, vec![Basic::negative_one(), Basic::half()]);
        let (re, im) = parts(&numeric_value(&minus_one_half, 53).unwrap());
        assert!(re.to_f64().value().abs() < 1e-15 && (im.to_f64().value() - 1.0).abs() < 1e-15);
        let s = Basic::new(
            Head::Add,
            vec![
                Basic::negative_one(),
                Basic::new(Head::Pow, vec![Basic::int(2), Basic::half()]),
            ],
        );
        assert!((parts(&numeric_value(&s, 53).unwrap()).0.to_f64().value() - (2f64.sqrt() - 1.0)).abs() < 1e-15);
        let one_i = Basic::new(Head::Add, vec![Basic::one(), Basic::imaginary_unit()]);
        let (re, im) = parts(&numeric_value(&Basic::new(Head::Pow, vec![one_i, Basic::int(2)]), 53).unwrap());
        assert!(re.to_f64().value().abs() < 1e-15 && (im.to_f64().value() - 2.0).abs() < 1e-15);
        assert!(numeric_value(&Basic::oo(), 53).is_none());
        assert!(numeric_value(&crate::core::Symbol::new("x", &[]).0, 53).is_none());
    }
}
