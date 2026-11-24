//! The formula tables of `sympy_extras/simplify/hyperexpand.py`
//! (`add_formulae`, `add_meijerg_formulae`), generated from the Python
//! tables: the parameters and the entries of `B`, `C`, `M` as printed,
//! the dummies `_a`, `_b`, `_c`, `_z`, `_rho` read back as the dummies of
//! the tables; and the matchers of the Meijer formulas.

#![allow(clippy::mutable_key_type)]

use super::hyperexpand::{Formula, GFunction, HyperFunction, Matcher, MeijerFormula};
use crate::core::basic::Basic;
use crate::error::PolyResult;
use crate::matrices::dense::ExprMatrix;

type Matrix = (usize, usize, &'static [&'static str]);

/// A formula: symbols, `ap`, `bq`, `B`, `C`, `M`.
const FORMULAE: &[(&[&str], &[&str], &[&str], Matrix, Matrix, Matrix)] = &[
    (&[], &[], &[], (1, 1, &["exp(_z)"]), (1, 1, &["1"]), (1, 1, &["_z"])),
    (&["_a"], &["_a"], &[], (1, 1, &["HyperRep_power1(-_a, _z)"]), (1, 1, &["1"]), (1, 1, &["_a*_z/(1 - _z)"])),
    (&["_a"], &["_a", "_a - 1/2"], &["2*_a"], (2, 1, &["HyperRep_power2(_a, _z)", "HyperRep_power2(_a + 1/2, _z)/2"]), (1, 2, &["1", "0"]), (2, 2, &["_z*(_a - 1/2)/(1 - _z)", "_z*(1/2 - _a)/(1 - _z)", "_a/(1 - _z)", "_a*(_z - 2)/(1 - _z)"])),
    (&[], &["1", "1"], &["2"], (2, 1, &["HyperRep_log1(_z)", "1"]), (1, 2, &["-1/_z", "0"]), (2, 2, &["0", "_z/(_z - 1)", "0", "0"])),
    (&[], &["1/2", "1"], &["3/2"], (2, 1, &["HyperRep_atanh(_z)", "1"]), (1, 2, &["1", "0"]), (2, 2, &["-1/2", "1/(2*(1 - _z))", "0", "0"])),
    (&[], &["1/2", "1/2"], &["3/2"], (2, 1, &["HyperRep_asin1(_z)", "HyperRep_power1(-1/2, _z)"]), (1, 2, &["1", "0"]), (2, 2, &["-1/2", "1/2", "0", "_z/(2*(1 - _z))"])),
    (&["_a"], &["_a", "_a + 1/2"], &["1/2"], (2, 1, &["HyperRep_sqrts1(-_a, _z)", "-HyperRep_sqrts2(-_a - 1/2, _z)"]), (1, 2, &["1", "0"]), (2, 2, &["0", "-_a", "_z*(-2*_a - 1)/(2*(1 - _z))", "-_z*(-2*_a - 1)/(1 - _z) + 1/2"])),
    (&["_a"], &["_a", "-_a"], &["1/2"], (2, 1, &["HyperRep_cosasin(_a, _z)", "HyperRep_sinasin(_a, _z)"]), (1, 2, &["1", "0"]), (2, 2, &["0", "-_a", "_a*_z/(1 - _z)", "1/(2*(1 - _z))"])),
    (&[], &["1", "1"], &["3/2"], (2, 1, &["HyperRep_asin2(_z)", "1"]), (1, 2, &["1", "0"]), (2, 2, &["(_z - 1/2)/(1 - _z)", "1/(2*(1 - _z))", "0", "0"])),
    (&[], &["1/2", "1/2"], &["1"], (2, 1, &["elliptic_k(_z)", "elliptic_e(_z)"]), (1, 2, &["2/pi", "0"]), (2, 2, &["-1/2", "-1/(2*_z - 2)", "-1/2", "1/2"])),
    (&[], &["-1/2", "1/2"], &["1"], (2, 1, &["elliptic_k(_z)", "elliptic_e(_z)"]), (1, 2, &["0", "2/pi"]), (2, 2, &["-1/2", "-1/(2*_z - 2)", "-1/2", "1/2"])),
    (&[], &["-1/2", "1", "1"], &["1/2", "2"], (3, 1, &["_z*HyperRep_atanh(_z)", "HyperRep_log1(_z)", "1"]), (1, 3, &["-2/3", "-1/(3*_z)", "2/3"]), (3, 3, &["1/2", "0", "_z/(2*(1 - _z))", "0", "0", "_z/(_z - 1)", "0", "0", "0"])),
    (&[], &["-1/2", "1", "1"], &["2", "2"], (3, 1, &["HyperRep_power1(1/2, _z)", "HyperRep_log2(_z)", "1"]), (1, 3, &["4/9 - 16/(9*_z)", "4/(3*_z)", "16/(9*_z)"]), (3, 3, &["_z/(2*(_z - 1))", "0", "0", "1/(2*_z - 2)", "0", "1/2", "0", "0", "0"])),
    (&["_b"], &["1"], &["_b"], (2, 1, &["_z**(1 - _b)*exp(_z)*lowergamma(_b - 1, _z)", "1"]), (1, 2, &["_b - 1", "0"]), (2, 2, &["-_b + _z + 1", "1", "0", "0"])),
    (&["_a"], &["_a"], &["2*_a"], (2, 1, &["4**(_a - 1/2)*_z**(1/2 - _a)*exp(_z/2)*besseli(_a - 1/2, _z/2)*gamma(_a + 1/2)", "4**(_a - 1/2)*_z**(1/2 - _a)*exp(_z/2)*besseli(_a + 1/2, _z/2)*gamma(_a + 1/2)"]), (1, 2, &["1", "0"]), (2, 2, &["_z/2", "_z/2", "_z/2", "-2*_a + _z/2"])),
    (&["_a"], &["_a"], &["_a + 1"], (2, 1, &["_a*lowergamma(_a, _z*exp_polar(I*pi))/(_z*exp_polar(I*pi))**_a", "_a*exp(_z)"]), (1, 2, &["1", "0"]), (2, 2, &["-_a", "1", "0", "_z"])),
    (&[], &["-1/2"], &["1/2"], (2, 1, &["sqrt(_z)*I*sqrt(pi)*erf(sqrt(_z)*I) + exp(_z)", "sqrt(_z)*I*sqrt(pi)*erf(sqrt(_z)*I)/2"]), (1, 2, &["1", "0"]), (2, 2, &["0", "1", "-_z/2", "_z + 1/2"])),
    (&[], &["1"], &["3/4", "5/4"], (3, 1, &["sqrt(pi)*(I*sinh(2*sqrt(_z))*fresnels(2*_z**(1/4)*exp(I*pi/4)/sqrt(pi)) + cosh(2*sqrt(_z))*fresnelc(2*_z**(1/4)*exp(I*pi/4)/sqrt(pi)))*exp(-I*pi/4)/(2*_z**(1/4))", "_z**(1/4)*sqrt(pi)*(sinh(2*sqrt(_z))*fresnelc(2*_z**(1/4)*exp(I*pi/4)/sqrt(pi)) + I*cosh(2*sqrt(_z))*fresnels(2*_z**(1/4)*exp(I*pi/4)/sqrt(pi)))*exp(-I*pi/4)/2", "1"]), (1, 3, &["1", "0", "0"]), (3, 3, &["-1/4", "1", "1/4", "_z", "1/4", "0", "0", "0", "0"])),
    (&["_a"], &["1/2", "_a"], &["3/2", "_a + 1"], (3, 1, &["-_a*I*sqrt(pi)*sqrt(1/_z)*erf(sqrt(_z)*I)/(2*_a - 1)", "_a*lowergamma(_a, _z*exp_polar(I*pi))/((_z*exp_polar(I*pi))**_a*(2*_a - 1))", "_a*exp(_z)/(2*_a - 1)"]), (1, 3, &["1", "-1", "0"]), (3, 3, &["-1/2", "0", "1", "0", "-_a", "1", "0", "0", "_z"])),
    (&[], &["1", "1"], &["2", "2"], (4, 1, &["-log(_z) + Ei(_z)", "exp(_z)", "1", "EulerGamma"]), (1, 4, &["1/_z", "0", "0", "-1/_z"]), (4, 4, &["0", "1", "-1", "0", "0", "_z", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"])),
    (&[], &[], &["1/2"], (2, 1, &["cosh(2*sqrt(_z))", "sqrt(_z)*sinh(2*sqrt(_z))"]), (1, 2, &["1", "0"]), (2, 2, &["0", "1", "_z", "1/2"])),
    (&["_b"], &[], &["_b"], (2, 1, &["_z**(1/2 - _b/2)*besseli(_b - 1, 2*sqrt(_z))*gamma(_b)", "_z**(1 - _b/2)*besseli(_b, 2*sqrt(_z))*gamma(_b)"]), (1, 2, &["1", "0"]), (2, 2, &["0", "1", "_z", "1 - _b"])),
    (&["_a"], &[], &["1/2", "_a", "_a + 1/2"], (4, 1, &["_z**(1/4 - _a/2)*(besseli(2*_a - 1, 4*_z**(1/4)) + besselj(2*_a - 1, 4*_z**(1/4)))*gamma(2*_a)/2**(2*_a)", "_z**(1/4)*_z**(1/4 - _a/2)*(besseli(2*_a, 4*_z**(1/4)) - besselj(2*_a, 4*_z**(1/4)))*gamma(2*_a)/2**(2*_a)", "sqrt(_z)*_z**(1/4 - _a/2)*(besseli(2*_a - 1, 4*_z**(1/4)) - besselj(2*_a - 1, 4*_z**(1/4)))*gamma(2*_a)/2**(2*_a)", "_z**(3/4)*_z**(1/4 - _a/2)*(besseli(2*_a, 4*_z**(1/4)) + besselj(2*_a, 4*_z**(1/4)))*gamma(2*_a)/2**(2*_a)"]), (1, 4, &["1", "0", "0", "0"]), (4, 4, &["0", "1", "0", "0", "0", "1/2 - _a", "1", "0", "0", "0", "1/2", "1", "_z", "0", "0", "1 - _a"])),
    (&["_a"], &[], &["_a", "_a + 1/2", "2*_a"], (4, 1, &["(2*sqrt(_z)*exp_polar(I*pi/2))**(1 - 2*_a)*besseli(2*_a - 1, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4))*besselj(2*_a - 1, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4))*gamma(2*_a)**2", "2*sqrt(2)*_z**(1/4)*(2*sqrt(_z)*exp_polar(I*pi/2))**(1 - 2*_a)*(besseli(2*_a, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4))*besselj(2*_a - 1, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4)) - besseli(2*_a - 1, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4))*besselj(2*_a, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4)))*exp_polar(I*pi/4)*gamma(2*_a)**2", "8*sqrt(_z)*(2*sqrt(_z)*exp_polar(I*pi/2))**(1 - 2*_a)*besseli(2*_a, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4))*besselj(2*_a, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4))*exp_polar(I*pi/2)*gamma(2*_a)**2", "16*sqrt(2)*_z**(3/4)*(2*sqrt(_z)*exp_polar(I*pi/2))**(1 - 2*_a)*(besseli(2*_a, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4))*besselj(2*_a - 1, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4)) + besseli(2*_a - 1, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4))*besselj(2*_a, 2*sqrt(2)*_z**(1/4)*exp_polar(I*pi/4)))*exp_polar(3*I*pi/4)*gamma(2*_a)**2"]), (1, 4, &["1", "0", "0", "0"]), (4, 4, &["0", "1/4", "0", "0", "0", "1/2 - _a", "-1/2", "0", "0", "0", "1 - 2*_a", "1/4", "-32*_z", "0", "0", "1 - _a"])),
    (&["_a"], &["_a"], &["_a - 1/2", "2*_a"], (3, 1, &["_z**(1/2 - _a)*besseli(_a - 1/2, sqrt(_z))**2", "_z**(1 - _a)*besseli(_a - 3/2, sqrt(_z))*besseli(_a - 1/2, sqrt(_z))", "_z**(3/2 - _a)*besseli(_a - 3/2, sqrt(_z))**2"]), (1, 3, &["-4**(_a - 1/2)*gamma(_a + 1/2)**2", "2*4**(_a - 1)*gamma(_a - 1/2)*gamma(_a + 1/2)", "0"]), (3, 3, &["1 - 2*_a", "1", "0", "_z/2", "1/2 - _a", "1/2", "0", "_z", "0"])),
    (&["_b"], &["1/2"], &["_b", "2 - _b"], (3, 1, &["pi*(1 - _b)*besseli(1 - _b, sqrt(_z))*besseli(_b - 1, sqrt(_z))/sin(_b*pi)", "sqrt(_z)*pi*(1 - _b)*(besseli(-_b, sqrt(_z))*besseli(_b - 1, sqrt(_z)) + besseli(_b, sqrt(_z))*besseli(1 - _b, sqrt(_z)))/sin(_b*pi)", "pi*(1 - _b)*besseli(-_b, sqrt(_z))*besseli(_b, sqrt(_z))/sin(_b*pi)"]), (1, 3, &["1", "0", "0"]), (3, 3, &["_b - 1", "1/2", "0", "_z", "0", "_z", "0", "1/2", "-_b"])),
    (&[], &["1/2"], &["3/2", "3/2"], (3, 1, &["Shi(2*sqrt(_z))/(2*sqrt(_z))", "sinh(2*sqrt(_z))/(2*sqrt(_z))", "cosh(2*sqrt(_z))"]), (1, 3, &["1", "0", "0"]), (3, 3, &["-1/2", "1/2", "0", "0", "-1/2", "1/2", "0", "2*_z", "0"])),
    (&[], &["3/4"], &["3/2", "7/4"], (3, 1, &["sqrt(pi)*exp(-3*I*pi/4)*fresnels(2*_z**(1/4)*exp(I*pi/4)/sqrt(pi))/(8*_z**(3/4))", "sinh(2*sqrt(_z))/sqrt(_z)", "cosh(2*sqrt(_z))"]), (1, 3, &["6", "0", "0"]), (3, 3, &["-3/4", "1/16", "0", "0", "-1/2", "1", "0", "_z", "0"])),
    (&[], &["1/4"], &["1/2", "5/4"], (3, 1, &["sqrt(pi)*exp(-I*pi/4)*fresnelc(2*_z**(1/4)*exp(I*pi/4)/sqrt(pi))/(2*_z**(1/4))", "cosh(2*sqrt(_z))", "sqrt(_z)*sinh(2*sqrt(_z))"]), (1, 3, &["1", "0", "0"]), (3, 3, &["-1/4", "1/4", "0", "0", "0", "1", "0", "_z", "1/2"])),
    (&["_a", "_b"], &["_a", "_a + 1/2"], &["2*_a", "_b", "2*_a - _b + 1"], (4, 1, &["(sqrt(_z)/2)**(1 - 2*_a)*besseli(2*_a - _b, sqrt(_z))*besseli(_b - 1, sqrt(_z))*gamma(_b)*gamma(2*_a - _b + 1)", "sqrt(_z)*(sqrt(_z)/2)**(1 - 2*_a)*besseli(_b, sqrt(_z))*besseli(2*_a - _b, sqrt(_z))*gamma(_b)*gamma(2*_a - _b + 1)", "sqrt(_z)*(sqrt(_z)/2)**(1 - 2*_a)*besseli(_b - 1, sqrt(_z))*besseli(2*_a - _b + 1, sqrt(_z))*gamma(_b)*gamma(2*_a - _b + 1)", "(sqrt(_z)/2)**(1 - 2*_a)*besseli(_b, sqrt(_z))*besseli(2*_a - _b + 1, sqrt(_z))*gamma(_b)*gamma(2*_a - _b + 1)"]), (1, 4, &["1", "0", "0", "0"]), (4, 4, &["0", "1/2", "1/2", "0", "_z/2", "1 - _b", "0", "_z/2", "_z/2", "0", "-2*_a + _b", "_z/2", "0", "1/2", "1/2", "-2*_a"])),
    (&[], &["1", "1"], &["2", "2", "3/2"], (5, 1, &["-log(2*sqrt(_z)) + Chi(2*sqrt(_z))", "cosh(2*sqrt(_z))", "sqrt(_z)*sinh(2*sqrt(_z))", "1", "EulerGamma"]), (1, 5, &["1/_z", "0", "0", "0", "-1/_z"]), (5, 5, &["0", "1/2", "0", "-1/2", "0", "0", "0", "1", "0", "0", "0", "_z", "1/2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"])),
    (&["_a"], &["1", "1", "_a"], &["2", "2", "_a + 1"], (4, 1, &["_a*(log(-_z) + expint(1, -_z) + EulerGamma)/(_z*(_a**2 - 2*_a + 1))", "_a*(gamma(_a) - uppergamma(_a, -_z))/((-_z)**_a*(_a - 1)**2)", "_a*exp(_z)/(_a**2 - 2*_a + 1)", "_a/(_z*(_a**2 - 2*_a + 1))"]), (1, 4, &["1 - _a", "1", "-1/_z", "1"]), (4, 4, &["-1", "0", "-1/_z", "1", "0", "-_a", "1", "0", "0", "0", "_z", "0", "0", "0", "0", "-1"])),
];

/// A Meijer formula: `an`, `ap`, `bm`, `bq`, `B`, `C`, `M`.
const MEIJER_FORMULAE: &[(&[&str], &[&str], &[&str], &[&str], Matrix, Matrix, Matrix)] = &[
    (
        &["_a + _rho"],
        &[],
        &["_rho", "_a + _rho"],
        &[],
        (
            2,
            1,
            &[
                "_z**_rho*exp(_z)*gamma(1 - _a)*uppergamma(_a, _z)",
                "_z**(_a + _rho)*gamma(1 - _a)",
            ],
        ),
        (1, 2, &["1", "0"]),
        (2, 2, &["_rho + _z", "-1", "0", "_a + _rho"]),
    ),
    (
        &["_a"],
        &[],
        &["_a", "_a", "_a - 1/2"],
        &[],
        (
            3,
            1,
            &[
                "_z**(_a - 1/2)*sqrt(pi)*((Si(2*sqrt(_z)) - pi/2)*cos(2*sqrt(_z)) - sin(2*sqrt(_z))*Ci(2*sqrt(_z)))",
                "_z**_a*sqrt(pi)*((Si(2*sqrt(_z)) - pi/2)*sin(2*sqrt(_z)) + cos(2*sqrt(_z))*Ci(2*sqrt(_z)))",
                "_z**_a*sqrt(pi)",
            ],
        ),
        (1, 3, &["-2", "0", "0"]),
        (3, 3, &["_a - 1/2", "-1", "0", "_z", "_a", "1/2", "0", "0", "_a"]),
    ),
    // not in SymPy's table (hyperexpand(meijerg([], [], [0, 0], [], z))
    // stays there, sympy/sympy#31017): G^{2,0}_{0,2}(z | rho, rho) =
    // 2*z**rho*K_0(2*sqrt(z)), DLMF 16.18.1 with 10.25
    (
        &[],
        &[],
        &["_rho", "_rho"],
        &[],
        (
            2,
            1,
            &[
                "_z**_rho*besselk(0, 2*sqrt(_z))",
                "_z**(_rho + 1/2)*besselk(1, 2*sqrt(_z))",
            ],
        ),
        (1, 2, &["2", "0"]),
        (2, 2, &["_rho", "-1", "-_z", "_rho"]),
    ),
];

struct Names {
    pairs: Vec<(Basic, Basic)>,
}

impl Names {
    fn new(names: &[&str]) -> Names {
        let pairs = names
            .iter()
            .map(|n| {
                (
                    crate::core::symbol::Symbol::new(&format!("_{}", n), &[]).0,
                    crate::core::symbol::Dummy::new(n, &[]).0,
                )
            })
            .collect();
        Names { pairs }
    }

    fn get(&self, name: &str) -> Basic {
        self.pairs
            .iter()
            .find(|(s, _)| s.to_string() == format!("_{}", name))
            .map(|(_, d)| d.clone())
            .expect("a table dummy")
    }

    fn read(&self, text: &str) -> PolyResult<Basic> {
        let parsed: Basic = crate::core::parsing::parse(text, true)?;
        let map: std::collections::HashMap<Basic, Basic> = self.pairs.iter().cloned().collect();
        crate::core::function::xreplace_evaluated(&parsed, &map)
    }

    fn list(&self, texts: &[&str]) -> PolyResult<Vec<Basic>> {
        texts.iter().map(|t| self.read(t)).collect()
    }

    fn matrix(&self, m: &Matrix) -> PolyResult<ExprMatrix> {
        ExprMatrix::new(m.0, m.1, self.list(m.2)?)
    }
}

/// The hypergeometric formulas (`add_formulae`).
pub(crate) fn add_formulae(formulae: &mut Vec<Formula>) -> PolyResult<()> {
    let names: Names = Names::new(&["a", "b", "c", "z"]);
    let z: Basic = names.get("z");
    for (symbols, ap, bq, b, c, m) in FORMULAE {
        let func = HyperFunction::new(&names.list(ap)?, &names.list(bq)?)?;
        let symbols: Vec<Basic> = symbols.iter().map(|s| names.read(s)).collect::<PolyResult<_>>()?;
        formulae.push(Formula {
            func,
            z: z.clone(),
            symbols,
            b: names.matrix(b)?,
            c: names.matrix(c)?,
            m: names.matrix(m)?,
        });
    }
    Ok(())
}

thread_local! {
    static MEIJER_NAMES: Names = Names::new(&["a", "b", "c", "z", "rho"]);
}

/// The Meijer formulas (`add_meijerg_formulae`).
pub(crate) fn add_meijerg_formulae(formulae: &mut Vec<MeijerFormula>) -> PolyResult<()> {
    MEIJER_NAMES.with(|names| -> PolyResult<()> {
        let z: Basic = names.get("z");
        let symbols: Vec<Basic> = ["a", "b", "c", "rho"].iter().map(|n| names.get(n)).collect();
        let matchers: [Matcher; 3] = [Matcher::UpperGamma, Matcher::ThreeOneOneThree, Matcher::BesselK];
        for ((an, ap, bm, bq, b, c, m), matcher) in MEIJER_FORMULAE.iter().zip(matchers) {
            let func = GFunction::new(&names.list(an)?, &names.list(ap)?, &names.list(bm)?, &names.list(bq)?)?;
            formulae.push(MeijerFormula::new(
                func,
                &z,
                &symbols,
                names.matrix(b)?,
                names.matrix(c)?,
                names.matrix(m)?,
                Some(matcher),
            ));
        }
        Ok(())
    })
}

fn simplify(e: &Basic) -> PolyResult<Basic> {
    crate::simplify::simplify::simplify(e, &crate::simplify::simplify::Options::default())
}

fn mod1_of_difference(x: &Basic, y: &Basic) -> PolyResult<Basic> {
    super::hyperexpand::mod1_pub(&simplify(&super::hyperexpand::s2(x, y)?)?)
}

fn is_zero_number(e: &Basic) -> bool {
    e.is_number_node() && *e == Basic::int(0)
}

fn truth_gt(a: &Basic, b: &Basic) -> PolyResult<bool> {
    super::hyperexpand::truth_pub(&crate::core::relational::gt(a, b)?)
}

/// `detect_uppergamma` of the first Meijer formula; `symbols` are `a, b, c, rho`.
pub(crate) fn detect_uppergamma(
    func: &GFunction,
    symbols: &[Basic],
) -> PolyResult<Option<(Vec<(Basic, Basic)>, GFunction)>> {
    let x: &Basic = &func.an[0];
    let (mut y, mut w) = (func.bm[0].clone(), func.bm[1].clone());
    let mut swapped: bool = false;
    if is_zero_number(&mod1_of_difference(x, &y)?) {
        swapped = true;
        std::mem::swap(&mut y, &mut w);
    }
    if !is_zero_number(&mod1_of_difference(x, &w)?) || truth_gt(&super::hyperexpand::s2(x, &w)?, &Basic::int(0))? {
        return Ok(None);
    }
    let bm: Vec<Basic> = if swapped {
        vec![x.clone(), y.clone()]
    } else {
        vec![y.clone(), x.clone()]
    };
    let (a, rho) = (&symbols[0], &symbols[3]);
    let repl: Vec<(Basic, Basic)> = vec![(rho.clone(), y.clone()), (a.clone(), super::hyperexpand::s2(x, &y)?)];
    Ok(Some((repl, GFunction::new(std::slice::from_ref(x), &[], &bm, &[])?)))
}

/// `detect_3113` of the second Meijer formula.
pub(crate) fn detect_3113(func: &GFunction, symbols: &[Basic]) -> PolyResult<Option<(Vec<(Basic, Basic)>, GFunction)>> {
    use super::hyperexpand::{a2, half, s2};
    let x: &Basic = &func.an[0];
    let (u, v, w) = (&func.bm[0], &func.bm[1], &func.bm[2]);
    let zero: Basic = Basic::int(0);
    let (sig, x1, x2, y): ([Basic; 3], &Basic, &Basic, &Basic) = if mod1_of_difference(u, v)? == zero {
        if mod1_of_difference(v, w)? == zero {
            return Ok(None);
        }
        ([half(), half(), zero.clone()], u, v, w)
    } else if mod1_of_difference(x, u)? == zero {
        ([half(), zero.clone(), half()], u, w, v)
    } else {
        ([zero.clone(), half(), half()], v, w, u)
    };
    if mod1_of_difference(x, x1)? != zero
        || mod1_of_difference(x, x2)? != zero
        || mod1_of_difference(x, y)? != half()
        || truth_gt(&s2(x, x1)?, &zero)?
        || truth_gt(&s2(x, x2)?, &zero)?
    {
        return Ok(None);
    }
    let shifted: Basic = s2(x, &half())?;
    let bm: Vec<Basic> = sig.iter().map(|t| a2(&shifted, t)).collect::<PolyResult<_>>()?;
    let repl: Vec<(Basic, Basic)> = vec![(symbols[0].clone(), x.clone())];
    Ok(Some((repl, GFunction::new(std::slice::from_ref(x), &[], &bm, &[])?)))
}

/// `detect_besselk` of the third Meijer formula: `G^{2,0}_{0,2}(z | b1, b2)`
/// with `b1 - b2` an integer, for which Slater's theorem has no formula
/// (the poles are double); the lower of the two parameters, the other
/// reached by raising it.
pub(crate) fn detect_besselk(
    func: &GFunction,
    symbols: &[Basic],
) -> PolyResult<Option<(Vec<(Basic, Basic)>, GFunction)>> {
    let (u, v) = (&func.bm[0], &func.bm[1]);
    let difference: Basic = simplify(&super::hyperexpand::s2(u, v)?)?;
    let Some(p) = difference.as_rational().filter(|r| r.is_int()) else {
        return Ok(None);
    };
    let y: Basic = if *p >= crate::numbers::Rat::from(0) {
        v.clone()
    } else {
        u.clone()
    };
    let repl: Vec<(Basic, Basic)> = vec![(symbols[3].clone(), y.clone())];
    Ok(Some((repl, GFunction::new(&[], &[], &[y.clone(), y], &[])?)))
}
