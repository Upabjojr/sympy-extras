//! The gamma function of a complex number at a given precision (what
//! mpmath's `gamma` gives the numerical values of the factorials): the
//! Stirling series of `log(gamma(w))` after shifting the argument to a
//! large real part, the reflection formula for a real part below 1/2.
//! `None` at the poles (the nonpositive integers).

use std::cell::RefCell;

use crate::core::evalf::{complex, complex_real, parts, real_int, real_rat, Complex, Real};
use crate::numbers::{bits, denom, numer, rat, Int, Rat};

thread_local! {
    /// `B_0, B_1, B_2, ...`, computed as far as needed.
    static BERNOULLI: RefCell<Vec<Rat>> = RefCell::new(vec![Rat::ONE]);
}

/// The Bernoulli number `B_m` (`B_1 = -1/2`).
fn bernoulli(m: usize) -> Rat {
    BERNOULLI.with(|cell| {
        let mut b = cell.borrow_mut();
        while b.len() <= m {
            let n: usize = b.len();
            // B_n = -1/(n + 1) * sum_{j < n} C(n + 1, j) B_j
            let mut total: Rat = Rat::ZERO;
            let mut c: Int = Int::ONE;
            for (j, bj) in b.iter().enumerate() {
                total += Rat::from(c.clone()) * bj;
                c = c * Int::from(n + 1 - j) / Int::from(j + 1);
            }
            let value: Rat = -total / Rat::from(Int::from(n + 1));
            b.push(value);
        }
        b[m].clone()
    })
}

fn log2_abs(r: &Rat) -> f64 {
    bits(&numer(r)) as f64 - bits(&denom(r)) as f64
}

fn to_f64(x: &Real) -> f64 {
    x.to_f64().value()
}

/// Whether `z` is a pole of gamma (a nonpositive integer).
fn is_pole(z: &Complex) -> bool {
    let (re, im) = parts(z);
    im == Real::ZERO && re <= Real::ZERO && re.clone() == re.floor()
}

fn c_int(n: i64, wp: usize) -> Complex {
    complex_real(real_int(&Int::from(n), wp), wp)
}

/// `gamma(z)` to `prec` bits, `None` at a pole.
pub fn gamma(z: &Complex, prec: usize) -> Option<Complex> {
    if is_pole(z) {
        return None;
    }
    let wp: usize = prec + 40;
    let (re, im) = parts(z);
    let real_argument: bool = im == Real::ZERO;
    let z: Complex = complex(re.clone().with_precision(wp).value(), im.with_precision(wp).value());
    let value: Complex = if re < real_rat(&rat(1, 2), wp) {
        // gamma(z) = pi / (sin(pi*z) * gamma(1 - z))
        let pi: Complex = complex_real(Real::pi(wp), wp);
        let one_minus: Complex = &c_int(1, wp) - &z;
        let g: Complex = gamma_shifted(&one_minus, wp)?;
        let pz: Complex = &pi * &z;
        let i: Complex = complex(real_int(&Int::ZERO, wp), real_int(&Int::ONE, wp));
        let (a, b) = ((&i * &pz).exp(), (-(&i * &pz)).exp());
        let sin: Complex = &(&a - &b) / &(&c_int(2, wp) * &i);
        &pi / &(&sin * &g)
    } else {
        gamma_shifted(&z, wp)?
    };
    let (vr, vi) = parts(&value);
    let vr: Real = vr.with_precision(prec).value();
    let vi: Real = if real_argument {
        real_int(&Int::ZERO, prec)
    } else {
        vi.with_precision(prec).value()
    };
    Some(complex(vr, vi))
}

/// `gamma(z)` for a real part at least 1/2: the Stirling series at
/// `w = z + N` and the recurrence back.
fn gamma_shifted(z: &Complex, wp: usize) -> Option<Complex> {
    let threshold: f64 = (wp as f64) * 0.2 + 10.0;
    let (re, _) = parts(z);
    let shift: i64 = (threshold - to_f64(&re)).ceil().max(0.0) as i64;
    let mut w: Complex = z.clone();
    let mut product: Complex = c_int(1, wp);
    for _ in 0..shift {
        product = &product * &w;
        w = &w + &c_int(1, wp);
    }
    let half: Complex = complex_real(real_rat(&rat(1, 2), wp), wp);
    let two_pi: Real = Real::pi(wp) * real_int(&Int::from(2u8), wp);
    let log_w: Complex = w.ln();
    let mut total: Complex = &(&(&w - &half) * &log_w) - &w;
    total = &total + &complex_real(two_pi.ln() * real_rat(&rat(1, 2), wp), wp);
    let (wr, wi) = parts(&w);
    let modulus: f64 = (to_f64(&wr).powi(2) + to_f64(&wi).powi(2)).sqrt();
    let w2: Complex = &w * &w;
    let mut power: Complex = w.clone();
    for k in 1..4 * wp {
        let b: Rat = bernoulli(2 * k);
        let scale: Rat = Rat::from(Int::from(2 * k)) * Rat::from(Int::from(2 * k - 1));
        let size: f64 = log2_abs(&b) - ((2 * k * (2 * k - 1)) as f64).log2() - (2 * k - 1) as f64 * modulus.log2();
        let coefficient: Complex = complex_real(real_rat(&(b / scale), wp), wp);
        total = &total + &(&coefficient / &power);
        if size < -(wp as f64) {
            break;
        }
        power = &power * &w2;
    }
    Some(&total.exp() / &product)
}

fn shift_for(z: &Complex, wp: usize) -> i64 {
    let threshold: f64 = (wp as f64) * 0.2 + 10.0;
    let (re, _) = parts(z);
    (threshold - to_f64(&re)).ceil().max(0.0) as i64
}

fn modulus(w: &Complex) -> f64 {
    let (wr, wi) = parts(w);
    (to_f64(&wr).powi(2) + to_f64(&wi).powi(2)).sqrt()
}

fn finish(value: &Complex, prec: usize, real: bool) -> Complex {
    let (vr, vi) = parts(value);
    let vi: Real = if real {
        real_int(&Int::ZERO, prec)
    } else {
        vi.with_precision(prec).value()
    };
    complex(vr.with_precision(prec).value(), vi)
}

/// `loggamma(z)` (mpmath's: the branch continuous from the positive real
/// axis, `loggamma(z + 1) = loggamma(z) + log(z)`) to `prec` bits, `None`
/// at a pole.
pub fn loggamma(z: &Complex, prec: usize) -> Option<Complex> {
    if is_pole(z) {
        return None;
    }
    let wp: usize = prec + 40;
    let (re, im) = parts(z);
    let positive_real: bool = im == Real::ZERO && re > Real::ZERO;
    let mut w: Complex = complex(re.with_precision(wp).value(), im.with_precision(wp).value());
    let mut logs: Complex = c_int(0, wp);
    for _ in 0..shift_for(&w, wp) {
        logs = &logs + &w.ln();
        w = &w + &c_int(1, wp);
    }
    let half: Complex = complex_real(real_rat(&rat(1, 2), wp), wp);
    let two_pi: Real = Real::pi(wp) * real_int(&Int::from(2u8), wp);
    let mut total: Complex = &(&(&w - &half) * &w.ln()) - &w;
    total = &total + &complex_real(two_pi.ln() * real_rat(&rat(1, 2), wp), wp);
    let size_w: f64 = modulus(&w);
    let w2: Complex = &w * &w;
    let mut power: Complex = w.clone();
    for k in 1..4 * wp {
        let b: Rat = bernoulli(2 * k);
        let scale: Rat = Rat::from(Int::from(2 * k)) * Rat::from(Int::from(2 * k - 1));
        let size: f64 = log2_abs(&b) - ((2 * k * (2 * k - 1)) as f64).log2() - (2 * k - 1) as f64 * size_w.log2();
        total = &total + &(&complex_real(real_rat(&(b / scale), wp), wp) / &power);
        if size < -(wp as f64) {
            break;
        }
        power = &power * &w2;
    }
    Some(finish(&(&total - &logs), prec, positive_real))
}

fn factorial_f(n: u64) -> Int {
    let mut r: Int = Int::ONE;
    for i in 2..=n {
        r *= Int::from(i);
    }
    r
}

/// `polygamma(m, z)` for an integer `m >= 0` to `prec` bits, `None` at a
/// pole: the asymptotic series at `w = z + N` and the recurrence
/// `polygamma(m, z + 1) = polygamma(m, z) + (-1)**m*m!/z**(m + 1)`.
pub fn polygamma(m: u64, z: &Complex, prec: usize) -> Option<Complex> {
    if is_pole(z) {
        return None;
    }
    let wp: usize = prec + 40 + 2 * m as usize;
    let (re, im) = parts(z);
    let real: bool = im == Real::ZERO;
    let z: Complex = complex(re.with_precision(wp).value(), im.with_precision(wp).value());
    let mut w: Complex = z.clone();
    let mut correction: Complex = c_int(0, wp);
    let shift: i64 = shift_for(&w, wp) + m as i64;
    for _ in 0..shift {
        correction = &correction + &(&c_int(1, wp) / &w.powi(Int::from(m + 1)));
        w = &w + &c_int(1, wp);
    }
    let mfact: Int = factorial_f(m);
    let size_w: f64 = modulus(&w);
    let total: Complex = if m == 0 {
        // log(w) - 1/(2w) - sum B_2k/(2k w**2k)
        let mut t: Complex = &w.ln() - &(&c_int(1, wp) / &(&c_int(2, wp) * &w));
        let w2: Complex = &w * &w;
        let mut power: Complex = w2.clone();
        for k in 1..4 * wp {
            let b: Rat = bernoulli(2 * k);
            let size: f64 = log2_abs(&b) - ((2 * k) as f64).log2() - (2 * k) as f64 * size_w.log2();
            let c: Rat = b / Rat::from(Int::from(2 * k));
            t = &t - &(&complex_real(real_rat(&c, wp), wp) / &power);
            if size < -(wp as f64) {
                break;
            }
            power = &power * &w2;
        }
        t
    } else {
        // (-1)**(m + 1) * [(m - 1)!/w**m + m!/(2 w**(m + 1)) + sum B_2k (2k + m - 1)!/((2k)! w**(2k + m))]
        let mut t: Complex = &complex_real(real_int(&factorial_f(m - 1), wp), wp) / &w.powi(Int::from(m));
        t = &t + &(&complex_real(real_int(&mfact, wp), wp) / &(&c_int(2, wp) * &w.powi(Int::from(m + 1))));
        let w2: Complex = &w * &w;
        let mut power: Complex = w.powi(Int::from(m + 2));
        for k in 1..4 * wp as u64 {
            let b: Rat = bernoulli(2 * k as usize);
            let c: Rat = b * Rat::from(factorial_f(2 * k + m - 1)) / Rat::from(factorial_f(2 * k));
            let size: f64 = log2_abs(&c) - (2 * k + m) as f64 * size_w.log2();
            t = &t + &(&complex_real(real_rat(&c, wp), wp) / &power);
            if size < -(wp as f64) {
                break;
            }
            power = &power * &w2;
        }
        if m % 2 == 0 {
            -t
        } else {
            t
        }
    };
    // polygamma(m, z) = polygamma(m, w) - (-1)**m m! sum 1/(z + j)**(m + 1)
    let factor: Complex = complex_real(real_int(&mfact, wp), wp);
    let back: Complex = &factor * &correction;
    let value: Complex = if m % 2 == 0 { &total - &back } else { &total + &back };
    Some(finish(&value, prec, real))
}
