//! The Mathieu functions, ported from
//! `sympy_extras.functions.special.mathieu_functions`: `mathieus`,
//! `mathieuc`, `mathieusprime` and `mathieucprime` of `(a, q, z)`, with the
//! trigonometric limits at `q = 0` (`mathieus(a, 0, z) = sin(sqrt(a)*z)`),
//! the parity in `z`, the derivatives in `z` and the conjugates. No
//! numerical values: neither SymPy nor mpmath has them.

// PORTED FROM SYMPY: sympy/functions/special/mathieu_functions.py (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

use crate::core::add::sub;
use crate::core::assumptions::{FactList, Predicate};
use crate::core::basic::{Basic, Function, Head};
use crate::core::expr::is_true;
use crate::core::mul::mul;
use crate::core::parameters::evaluate;
use crate::core::power::pow;
use crate::core::printing::could_extract_minus_sign;
use crate::error::{not_ported, PolyResult};
use crate::functions::elementary::trigonometric::{cos, sin};

fn node(f: Function, args: Vec<Basic>) -> Basic {
    Basic::new(Head::Function(f), args)
}

fn neg(e: &Basic) -> PolyResult<Basic> {
    mul(&[Basic::negative_one(), e.clone()])
}

/// The Mathieu functions.
pub fn is_mathieu(f: Function) -> bool {
    matches!(
        f,
        Function::MathieuS | Function::MathieuC | Function::MathieuSPrime | Function::MathieuCPrime
    )
}

/// The evaluating function of the Mathieu functions.
pub fn mathieu(f: Function, args: &[Basic]) -> PolyResult<Basic> {
    if !evaluate() {
        return Ok(node(f, args.to_vec()));
    }
    let (a, q, z) = (&args[0], &args[1], &args[2]);
    if q.is_number_class() && is_true(q, Predicate::zero) {
        let root: Basic = pow(a, &Basic::half())?;
        let w: Basic = mul(&[root.clone(), z.clone()])?;
        return match f {
            Function::MathieuS => sin(&w),
            Function::MathieuC => cos(&w),
            Function::MathieuSPrime => mul(&[root, cos(&w)?]),
            _ => mul(&[Basic::negative_one(), root, sin(&w)?]),
        };
    }
    if could_extract_minus_sign(z) {
        let inner: Basic = mathieu(f, &[a.clone(), q.clone(), neg(z)?])?;
        return match f {
            Function::MathieuS | Function::MathieuCPrime => neg(&inner),
            _ => Ok(inner),
        };
    }
    Ok(node(f, args.to_vec()))
}

/// The derivative in the argument `argindex` (from 1, `fdiff`).
pub fn mathieu_fdiff(f: Function, e: &Basic, argindex: usize) -> PolyResult<Basic> {
    let args: &[Basic] = e.args();
    if argindex == 3 {
        let (a, q, z) = (&args[0], &args[1], &args[2]);
        // 2*q*cos(2*z) - a
        let factor = || -> PolyResult<Basic> {
            sub(
                &mul(&[
                    Basic::integer(2.into()),
                    q.clone(),
                    cos(&mul(&[Basic::integer(2.into()), z.clone()])?)?,
                ])?,
                a,
            )
        };
        return match f {
            Function::MathieuS => mathieu(Function::MathieuSPrime, args),
            Function::MathieuC => mathieu(Function::MathieuCPrime, args),
            Function::MathieuSPrime => mul(&[factor()?, mathieu(Function::MathieuS, args)?]),
            _ => mul(&[factor()?, mathieu(Function::MathieuC, args)?]),
        };
    }
    let v: &Basic = &args[argindex - 1];
    let elsewhere: bool = args
        .iter()
        .enumerate()
        .any(|(k, w)| k != argindex - 1 && w.free_symbols().contains(v));
    if v.is_symbol() && !elsewhere {
        return Ok(crate::core::function::derivative_node(e, &[(v.clone(), 1)]));
    }
    Err(not_ported("a derivative as a Subs"))
}

/// The conjugate (`MathieuBase._eval_conjugate`): of every argument.
pub fn mathieu_conjugate(f: Function, e: &Basic) -> PolyResult<Option<Basic>> {
    use crate::functions::elementary::complexes::conjugate;
    let mut args: Vec<Basic> = Vec::new();
    for a in e.args() {
        args.push(conjugate(a)?);
    }
    Ok(Some(mathieu(f, &args)?))
}

/// The facts (`_eval_facts`).
pub fn mathieu_facts() -> FactList {
    let mut facts: FactList = FactList::new();
    facts.set(Predicate::commutative, true);
    facts
}
