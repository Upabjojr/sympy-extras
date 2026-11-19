//! Ports of mpmath's `identification.pslq` and of its `polyroots`
//! (Durand–Kerner) for the structure theorem: the integer relations of
//! real numbers given at a working precision, in mpmath's fixed-point
//! arithmetic step for step, and the complex roots of a polynomial.

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use dashu_int::IBig;

use crate::core::evalf::{Complex, Real};

fn floor_div(a: &IBig, b: &IBig) -> IBig {
    let q: IBig = a / b;
    let r: IBig = a - &q * b;
    if r != IBig::ZERO && ((r < IBig::ZERO) != (*b < IBig::ZERO)) {
        q - IBig::ONE
    } else {
        q
    }
}

fn shr(a: &IBig, k: usize) -> IBig {
    // Python's arithmetic shift: the floor of a / 2**k
    floor_div(a, &(IBig::ONE << k))
}

fn abs(a: &IBig) -> IBig {
    if *a < IBig::ZERO {
        -a.clone()
    } else {
        a.clone()
    }
}

fn isqrt(a: &IBig) -> IBig {
    // the floor of the square root of a nonnegative integer
    if *a <= IBig::ZERO {
        return IBig::ZERO;
    }
    crate::numbers::isqrt(a)
}

/// `round_fixed`.
fn round_fixed(x: &IBig, prec: usize) -> IBig {
    shr(&(x + (IBig::ONE << (prec - 1))), prec) << prec
}

/// `sqrt_fixed`: `isqrt(x << prec)`.
fn sqrt_fixed(x: &IBig, prec: usize) -> IBig {
    isqrt(&(x << prec))
}

/// `to_fixed`: the binary float `x` as `floor(x * 2**prec)`.
pub fn to_fixed(x: &Real, prec: usize) -> IBig {
    let repr = x.repr();
    let man: IBig = repr.significand().clone();
    let offset: isize = repr.exponent() + prec as isize;
    if offset >= 0 {
        man << offset as usize
    } else {
        shr(&man, (-offset) as usize)
    }
}

/// mpmath's `pslq(x, maxcoeff=maxcoeff, maxsteps=maxsteps)` at the working
/// precision `prec` (bits): an integer relation of the values, or `None`.
pub fn pslq(values: &[Real], prec: usize, maxcoeff: i64, maxsteps: usize) -> Option<Vec<i64>> {
    let n: usize = values.len();
    if n < 2 || prec < 53 {
        return None;
    }
    let target: usize = (prec as f64 * 0.75) as usize;
    let prec: usize = prec + 60;
    // tol = 2**-target as a fixed-point number
    let tol: IBig = if prec >= target {
        IBig::ONE << (prec - target)
    } else {
        IBig::ZERO
    };
    if tol == IBig::ZERO {
        return None;
    }
    let mut x: Vec<IBig> = vec![IBig::ZERO];
    for v in values {
        x.push(to_fixed(v, prec));
    }
    let minx: IBig = x[1..].iter().map(abs).min().expect("values");
    if minx == IBig::ZERO {
        return None;
    }
    if minx < floor_div(&tol, &IBig::from(100)) {
        return None;
    }
    let one: IBig = IBig::ONE << prec;
    let g: IBig = sqrt_fixed(&floor_div(&(IBig::from(4) << prec), &IBig::from(3)), prec);
    let mut a: HashMap<(usize, usize), IBig> = HashMap::new();
    let mut b: HashMap<(usize, usize), IBig> = HashMap::new();
    let mut h: HashMap<(usize, usize), IBig> = HashMap::new();
    for i in 1..=n {
        for j in 1..=n {
            let v: IBig = if i == j { one.clone() } else { IBig::ZERO };
            a.insert((i, j), v.clone());
            b.insert((i, j), v);
            h.insert((i, j), IBig::ZERO);
        }
    }
    let mut s: Vec<IBig> = vec![IBig::ZERO; n + 1];
    for k in 1..=n {
        let mut t: IBig = IBig::ZERO;
        for xj in &x[k..=n] {
            t += shr(&(xj * xj), prec);
        }
        s[k] = sqrt_fixed(&t, prec);
    }
    let t: IBig = s[1].clone();
    let mut y: Vec<IBig> = x.clone();
    for k in 1..=n {
        y[k] = floor_div(&(&x[k] << prec), &t);
        s[k] = floor_div(&(&s[k] << prec), &t);
    }
    for i in 1..=n {
        for j in i + 1..n {
            h.insert((i, j), IBig::ZERO);
        }
        if i < n {
            let v: IBig = if s[i] != IBig::ZERO {
                floor_div(&(&s[i + 1] << prec), &s[i])
            } else {
                IBig::ZERO
            };
            h.insert((i, i), v);
        }
        for j in 1..i {
            let sjj1: IBig = &s[j] * &s[j + 1];
            let v: IBig = if sjj1 != IBig::ZERO {
                floor_div(&((-(&y[i] * &y[j])) << prec), &sjj1)
            } else {
                IBig::ZERO
            };
            h.insert((i, j), v);
        }
    }
    let get = |m: &HashMap<(usize, usize), IBig>, i: usize, j: usize| -> IBig { m[&(i, j)].clone() };
    for i in 2..=n {
        for j in (1..i).rev() {
            let hjj: IBig = get(&h, j, j);
            if hjj == IBig::ZERO {
                continue;
            }
            let t: IBig = round_fixed(&floor_div(&(get(&h, i, j) << prec), &hjj), prec);
            y[j] = &y[j] + shr(&(&t * &y[i]), prec);
            for k in 1..=j {
                let v: IBig = get(&h, i, k) - shr(&(&t * get(&h, j, k)), prec);
                h.insert((i, k), v);
            }
            for k in 1..=n {
                let v: IBig = get(&a, i, k) - shr(&(&t * get(&a, j, k)), prec);
                a.insert((i, k), v);
                let w: IBig = get(&b, k, j) + shr(&(&t * get(&b, k, i)), prec);
                b.insert((k, j), w);
            }
        }
    }
    let maxc: IBig = IBig::from(maxcoeff);
    for _ in 0..maxsteps {
        let mut m: usize = 0;
        let mut szmax: IBig = IBig::from(-1);
        let mut gi: IBig = IBig::ONE;
        for i in 1..n {
            gi = &gi * &g;
            let sz: IBig = shr(&(&gi * abs(&get(&h, i, i))), prec * (i - 1));
            if sz > szmax {
                m = i;
                szmax = sz;
            }
        }
        y.swap(m, m + 1);
        for i in 1..=n {
            let (p, q) = (get(&h, m, i), get(&h, m + 1, i));
            h.insert((m, i), q);
            h.insert((m + 1, i), p);
            let (p, q) = (get(&a, m, i), get(&a, m + 1, i));
            a.insert((m, i), q);
            a.insert((m + 1, i), p);
            let (p, q) = (get(&b, i, m), get(&b, i, m + 1));
            b.insert((i, m), q);
            b.insert((i, m + 1), p);
        }
        if m + 2 <= n {
            let (hmm, hmm1) = (get(&h, m, m), get(&h, m, m + 1));
            let t0: IBig = sqrt_fixed(&shr(&(&hmm * &hmm + &hmm1 * &hmm1), prec), prec);
            if t0 == IBig::ZERO {
                break;
            }
            let t1: IBig = floor_div(&(&hmm << prec), &t0);
            let t2: IBig = floor_div(&(&hmm1 << prec), &t0);
            for i in m..=n {
                let (t3, t4) = (get(&h, i, m), get(&h, i, m + 1));
                h.insert((i, m), shr(&(&t1 * &t3 + &t2 * &t4), prec));
                h.insert((i, m + 1), shr(&(-(&t2 * &t3) + &t1 * &t4), prec));
            }
        }
        for i in m + 1..=n {
            let top: usize = (i - 1).min(m + 1);
            for j in (1..=top).rev() {
                let hjj: IBig = get(&h, j, j);
                if hjj == IBig::ZERO {
                    break;
                }
                let t: IBig = round_fixed(&floor_div(&(get(&h, i, j) << prec), &hjj), prec);
                y[j] = &y[j] + shr(&(&t * &y[i]), prec);
                for k in 1..=j {
                    let v: IBig = get(&h, i, k) - shr(&(&t * get(&h, j, k)), prec);
                    h.insert((i, k), v);
                }
                for k in 1..=n {
                    let v: IBig = get(&a, i, k) - shr(&(&t * get(&a, j, k)), prec);
                    a.insert((i, k), v);
                    let w: IBig = get(&b, k, j) + shr(&(&t * get(&b, k, i)), prec);
                    b.insert((k, j), w);
                }
            }
        }
        for i in 1..=n {
            let err: IBig = abs(&y[i]);
            if err < tol {
                let vec: Vec<IBig> = (1..=n).map(|j| shr(&round_fixed(&get(&b, j, i), prec), prec)).collect();
                if vec.iter().map(abs).max().is_some_and(|v| v < maxc) {
                    return Some(
                        vec.iter()
                            .map(|v| i64::try_from(v).expect("a small coefficient"))
                            .collect(),
                    );
                }
            }
        }
        let recnorm: IBig = h.values().map(abs).max().unwrap_or(IBig::ZERO);
        if recnorm != IBig::ZERO {
            let norm: IBig = floor_div(
                &shr(&floor_div(&(IBig::ONE << (2 * prec)), &recnorm), prec),
                &IBig::from(100),
            );
            if norm >= maxc {
                break;
            }
        } else {
            break;
        }
    }
    None
}

/// The complex roots of a polynomial with integer coefficients (highest
/// degree first) by the Durand–Kerner iteration of mpmath's `polyroots`, at
/// `prec` bits; `None` when it does not converge in `maxsteps`.
pub fn polyroots(coefficients: &[crate::numbers::Int], prec: usize, maxsteps: usize) -> Option<Vec<Complex>> {
    use crate::core::evalf::{complex, real_rat};
    use crate::numbers::Rat;
    let deg: usize = coefficients.len().checked_sub(1)?;
    if deg == 0 {
        return Some(vec![]);
    }
    let wp: usize = prec + 20;
    let lead: Rat = Rat::from(coefficients[0].clone());
    let monic: Vec<Complex> = coefficients
        .iter()
        .map(|c| complex(real_rat(&(Rat::from(c.clone()) / &lead), wp), real_rat(&Rat::ZERO, wp)))
        .collect();
    let eval = |z: &Complex| -> Complex {
        let mut r: Complex = monic[0].clone();
        for c in &monic[1..] {
            r = &r * z + c;
        }
        r
    };
    let seed: Complex = complex(
        real_rat(
            &Rat::from_parts(crate::numbers::Int::from(4), dashu_int::UBig::from(10u8)),
            wp,
        ),
        real_rat(
            &Rat::from_parts(crate::numbers::Int::from(9), dashu_int::UBig::from(10u8)),
            wp,
        ),
    );
    let mut roots: Vec<Complex> = Vec::new();
    let mut power: Complex = complex(real_rat(&Rat::ONE, wp), real_rat(&Rat::ZERO, wp));
    for _ in 0..deg {
        roots.push(power.clone());
        power = &power * &seed;
    }
    let tiny: Real = real_rat(
        &Rat::from_parts(crate::numbers::Int::ONE, dashu_int::UBig::ONE << prec),
        wp,
    );
    let size = |z: &Complex| -> Real {
        let (a, b) = crate::core::evalf::parts(z);
        (&a * &a + &b * &b).sqrt()
    };
    for _ in 0..maxsteps {
        let mut moved: Real = real_rat(&Rat::ZERO, wp);
        for i in 0..deg {
            let mut denominator: Complex = complex(real_rat(&Rat::ONE, wp), real_rat(&Rat::ZERO, wp));
            for j in 0..deg {
                if i != j {
                    denominator = &denominator * &(&roots[i] - &roots[j]);
                }
            }
            if size(&denominator) == real_rat(&Rat::ZERO, wp) {
                return None;
            }
            let step: Complex = eval(&roots[i]) / &denominator;
            let s: Real = size(&step);
            if s > moved {
                moved = s;
            }
            roots[i] = &roots[i] - &step;
        }
        if moved < tiny {
            return Some(roots);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::evalf::real_rat;
    use crate::numbers::Rat;

    fn log(n: u32, prec: usize) -> Real {
        real_rat(&Rat::from(crate::numbers::Int::from(n)), prec).ln()
    }

    #[test]
    fn relations_as_mpmath_finds_them() {
        let p: usize = 269;
        assert_eq!(
            pslq(&[log(2, p), log(3, p), log(6, p)], p, 10_000, 100_000),
            Some(vec![1, 1, -1])
        );
        let t: Real = real_rat(&Rat::from(crate::numbers::Int::from(3)), p).sqrt()
            + real_rat(
                &Rat::from_parts(crate::numbers::Int::ONE, dashu_int::UBig::from(7u8)),
                p,
            );
        let two_pi_t: Real = Real::pi(p) * real_rat(&Rat::from(crate::numbers::Int::from(2)), p) * &t;
        assert_eq!(
            pslq(&[log(12, p), log(2, p), log(3, p), two_pi_t], p, 10_000, 100_000),
            Some(vec![1, -2, -1, 0])
        );
    }

    #[test]
    fn durand_kerner() {
        use crate::numbers::Int;
        let roots = polyroots(&[Int::from(1), Int::from(0), Int::from(-2)], 100, 200).expect("converges");
        assert_eq!(roots.len(), 2);
    }
}
