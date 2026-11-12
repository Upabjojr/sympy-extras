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
pub(crate) fn bernoulli(m: usize) -> Rat {
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

fn c_pow(base: &Complex, s: &Complex) -> Complex {
    (s * &base.ln()).exp()
}

/// The Hurwitz zeta `zeta(s, a)` to `prec` bits (the Riemann zeta for
/// `a = 1`) by Euler-Maclaurin summation; `None` at `s = 1` and at a
/// nonpositive integer `a`.
pub fn hurwitz_zeta(s: &Complex, a: &Complex, prec: usize) -> Option<Complex> {
    let (sr, si) = parts(s);
    if si == Real::ZERO && sr == real_int(&Int::ONE, prec) {
        return None;
    }
    if is_pole(a) {
        return hurwitz_at_nonpositive_integer(s, a, prec);
    }
    let (ar, ai) = parts(a);
    let real: bool = si == Real::ZERO && ai == Real::ZERO && ar > Real::ZERO;
    let s_size: f64 = (to_f64(&sr).powi(2) + to_f64(&si).powi(2)).sqrt();
    let wp: usize = prec + 40 + (s_size.log2().max(0.0) as usize) * 4;
    let s: Complex = complex(sr.with_precision(wp).value(), si.with_precision(wp).value());
    let a: Complex = complex(ar.clone().with_precision(wp).value(), ai.with_precision(wp).value());
    let n: i64 = (wp as f64 * 0.25 + s_size + 10.0 - to_f64(&ar).min(0.0)).ceil() as i64;
    let minus_s: Complex = -&s;
    let mut total: Complex = c_int(0, wp);
    for k in 0..n {
        total = &total + &c_pow(&(&a + &c_int(k, wp)), &minus_s);
    }
    let an: Complex = &a + &c_int(n, wp);
    let one: Complex = c_int(1, wp);
    total = &total + &(&c_pow(&an, &(&one - &s)) / &(&s - &one));
    let an_minus_s: Complex = c_pow(&an, &minus_s);
    total = &total + &(&an_minus_s * &complex_real(real_rat(&rat(1, 2), wp), wp));
    // sum_j B_2j/(2j)! * s(s+1)...(s+2j-2) * an**(-s-2j+1)
    let an_size: f64 = modulus(&an);
    let mut rising: Complex = s.clone();
    let mut power: Complex = &an_minus_s / &an;
    let inverse_square: Complex = &one / &(&an * &an);
    let mut factorial: Int = Int::from(2u8);
    let mut magnitude: f64 = s_size.max(1.0).log2();
    for j in 1..4 * wp {
        let b: Rat = bernoulli(2 * j);
        let coefficient: Rat = b / Rat::from(factorial.clone());
        let term: Complex = &(&complex_real(real_rat(&coefficient, wp), wp) * &rising) * &power;
        total = &total + &term;
        let size: f64 = log2_abs(&coefficient) + magnitude - (2 * j) as f64 * an_size.log2();
        if size < -(wp as f64) {
            break;
        }
        // the next rising factorial s(s+1)...(s+2j)
        let next1: Complex = &s + &c_int(2 * j as i64 - 1, wp);
        let next2: Complex = &s + &c_int(2 * j as i64, wp);
        magnitude += modulus(&next1).max(1.0).log2() + modulus(&next2).max(1.0).log2();
        rising = &(&rising * &next1) * &next2;
        power = &power * &inverse_square;
        factorial = factorial * Int::from(2 * j + 1) * Int::from(2 * j + 2);
    }
    Some(finish(&total, prec, real))
}

/// `polylog(s, z)` by its series `sum z**k/k**s` for `|z| < 3/4`; `None`
/// elsewhere (mpmath's other formulas are not ported).
pub fn polylog_series(s: &Complex, z: &Complex, prec: usize) -> Option<Complex> {
    let size: f64 = modulus(z);
    if size >= 0.75 {
        return None;
    }
    let (sr, si) = parts(s);
    let (zr, zi) = parts(z);
    let real: bool = si == Real::ZERO && zi == Real::ZERO;
    let wp: usize = prec + 40;
    if size == 0.0 {
        return Some(complex(real_int(&Int::ZERO, prec), real_int(&Int::ZERO, prec)));
    }
    let re_s: f64 = to_f64(&sr);
    let s: Complex = complex(sr.with_precision(wp).value(), si.with_precision(wp).value());
    let z: Complex = complex(zr.with_precision(wp).value(), zi.with_precision(wp).value());
    let minus_s: Complex = -&s;
    let mut total: Complex = c_int(0, wp);
    let mut power: Complex = z.clone();
    for k in 1..(100 * wp as i64) {
        let term: Complex = &power * &c_pow(&c_int(k, wp), &minus_s);
        total = &total + &term;
        let estimate: f64 = k as f64 * size.log2() - re_s * (k as f64).log2();
        if estimate < -((wp + 10) as f64) && k > 2 {
            break;
        }
        power = &power * &z;
    }
    Some(finish(&total, prec, real))
}

/// `zeta(s, -m)` for an integer `m >= 0` as mpmath extends the sum: the
/// terms `(-j)**(-s)` on the principal branch, and the term of `0`: zero
/// for `re(-s) > 0`, one for `re(-s) = 0`, a pole otherwise.
fn hurwitz_at_nonpositive_integer(s: &Complex, a: &Complex, prec: usize) -> Option<Complex> {
    let (sr, si) = parts(s);
    if sr > Real::ZERO {
        return None;
    }
    let wp: usize = prec + 40;
    let m: i64 = (-parts(a).0).to_int().value().try_into().ok()?;
    if m > 0 && si != Real::ZERO {
        // mpmath's fast sum for a complex s takes the log of a negative
        // integer there (ComplexResult): no value
        return None;
    }
    let s_wp: Complex = complex(sr.clone().with_precision(wp).value(), si.with_precision(wp).value());
    let minus_s: Complex = -&s_wp;
    let mut total: Complex = hurwitz_zeta(&s_wp, &c_int(1, wp), wp)?;
    if sr == Real::ZERO {
        total = &total + &c_int(1, wp);
    }
    for j in 1..=m {
        total = &total + &c_pow(&c_int(-j, wp), &minus_s);
    }
    Some(finish(&total, prec, false))
}

fn raise_to(z: &Complex, wp: usize) -> Complex {
    let (re, im) = parts(z);
    complex(re.with_precision(wp).value(), im.with_precision(wp).value())
}

fn is_zero_complex(z: &Complex) -> bool {
    let (re, im) = parts(z);
    re == Real::ZERO && im == Real::ZERO
}

/// `sum_k x**k/(a(a+1)...(a+k))`, the series of the lower incomplete
/// gamma.
fn lower_series(a: &Complex, x: &Complex, wp: usize) -> Complex {
    let mut term: Complex = &c_int(1, wp) / a;
    let mut total: Complex = term.clone();
    let mut k: i64 = 1;
    loop {
        term = &(&term * x) / &(a + &c_int(k, wp));
        total = &total + &term;
        let size: f64 = modulus(&term);
        if size == 0.0 || (k > 2 && size.log2() < -((wp + 10) as f64) && modulus(x) < k as f64) || k > 400_000 {
            break;
        }
        k += 1;
    }
    total
}

/// mpmath's `gammainc(a, 0, x)`, the lower incomplete gamma; `None` at a
/// pole of `gamma(a)`.
pub fn lower_gamma(a: &Complex, x: &Complex, prec: usize) -> Option<Complex> {
    if is_zero_complex(x) {
        return Some(complex(real_int(&Int::ZERO, prec), real_int(&Int::ZERO, prec)));
    }
    if is_pole(a) || modulus(x) > 2000.0 {
        return None;
    }
    let real: bool = parts(a).1 == Real::ZERO && parts(x).1 == Real::ZERO && parts(x).0 > Real::ZERO;
    let wp: usize = prec + 40 + (modulus(x) * 1.443) as usize;
    let (a, x) = (raise_to(a, wp), raise_to(x, wp));
    let power: Complex = (&a * &x.ln()).exp();
    let value: Complex = &(&power * &(-&x).exp()) * &lower_series(&a, &x, wp);
    Some(finish(&value, prec, real))
}

/// mpmath's `gammainc(a, x)`, the upper incomplete gamma: `gamma(a)` minus
/// the lower one, and at a pole `a = -N` of gamma the form
/// `(-1)**N/N!*(E1(x) - exp(-x)*sum_{k<N} (-1)**k*k!/x**(k+1))`.
pub fn upper_gamma(a: &Complex, x: &Complex, prec: usize) -> Option<Complex> {
    let (ar, ai) = parts(a);
    if is_zero_complex(x) {
        if ar > Real::ZERO {
            return gamma(a, prec);
        }
        return None;
    }
    if modulus(x) > 2000.0 {
        return None;
    }
    let real: bool = ai == Real::ZERO && parts(x).1 == Real::ZERO && parts(x).0 > Real::ZERO;
    let wp: usize = prec + 40 + (modulus(x) * 1.443) as usize * 2;
    let (aa, xx) = (raise_to(a, wp), raise_to(x, wp));
    let value: Complex = if is_pole(a) {
        let n: u64 = (-ar).to_int().value().try_into().ok()?;
        // E1(x) = -EulerGamma - log(x) - sum (-x)**k/(k*k!)
        let minus_x: Complex = -&xx;
        let mut term: Complex = c_int(1, wp);
        let mut series: Complex = c_int(0, wp);
        let mut k: i64 = 1;
        loop {
            term = &(&term * &minus_x) / &c_int(k, wp);
            let t: Complex = &term / &c_int(k, wp);
            series = &series + &t;
            let size: f64 = modulus(&t);
            if size == 0.0 || (k > 2 && size.log2() < -((wp + 10) as f64) && modulus(&xx) < k as f64) || k > 400_000 {
                break;
            }
            k += 1;
        }
        let g: Complex = complex_real(crate::core::evalf::euler_gamma(wp), wp);
        let e1: Complex = &(&(-&g) - &xx.ln()) - &series;
        let mut sum: Complex = c_int(0, wp);
        let mut factorial: Int = Int::ONE;
        let mut power: Complex = xx.clone();
        for k in 0..n {
            if k > 0 {
                factorial *= Int::from(k);
                power = &power * &xx;
            }
            let t: Complex = &complex_real(real_int(&factorial, wp), wp) / &power;
            sum = if k % 2 == 0 { &sum + &t } else { &sum - &t };
        }
        let mut nf: Int = Int::ONE;
        for k in 2..=n {
            nf *= Int::from(k);
        }
        let sign: i64 = if n % 2 == 1 { -1 } else { 1 };
        let inner: Complex = &e1 - &(&(-&xx).exp() * &sum);
        &inner * &complex_real(real_int(&Int::from(sign), wp) / real_int(&nf, wp), wp)
    } else {
        let lower: Complex = lower_gamma(&aa, &xx, wp)?;
        &gamma(&aa, wp)? - &lower
    };
    Some(finish(&value, prec, real))
}
