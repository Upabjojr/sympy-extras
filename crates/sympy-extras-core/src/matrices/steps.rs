//! Port of `sympy_extras/matrices/steps.py`: Gaussian and Gauss–Jordan
//! elimination step by step (the row operations with the matrix after
//! each), for echelon forms, reduced forms, inverses, linear systems and
//! determinants, in the field of the entries.

use crate::core::basic::Basic;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;
use crate::matrices::eigen::{Field, FunctionEntries, GaussianEntries, RationalEntries};
use crate::matrices::linalg::{complex_lift, kernels as kernels_of, normal, square, PolynomialLift};
use crate::polys::domains::domain::Domain;

fn int(n: i64) -> Basic {
    Basic::int(n)
}

/// The kind of an elementary row operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Swap,
    Scale,
    Add,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Swap => "swap",
            Kind::Scale => "scale",
            Kind::Add => "add",
        }
    }
}

/// An elementary row operation (`RowOperation`).
#[derive(Debug, Clone, PartialEq)]
pub struct RowOperation {
    pub kind: Kind,
    pub target: usize,
    pub source: usize,
    pub factor: Basic,
}

impl RowOperation {
    /// `R1 <-> R2`, `R1 <- (c)*R1`, `R2 <- R2 - (c)*R1` (`__repr__`).
    pub fn repr(&self) -> PolyResult<String> {
        let (target, source) = (format!("R{}", self.target + 1), format!("R{}", self.source + 1));
        Ok(match self.kind {
            Kind::Swap => format!("{} <-> {}", target, source),
            Kind::Scale => format!("{} <- ({})*{}", target, self.factor, target),
            Kind::Add => {
                let factor: &Basic = &self.factor;
                if *factor == int(1) {
                    format!("{} <- {} + {}", target, target, source)
                } else if *factor == int(-1) {
                    format!("{} <- {} - {}", target, target, source)
                } else {
                    let negative_rational: bool = factor.as_rational().is_some_and(crate::numbers::rat_is_negative);
                    let negative: bool = negative_rational || (!factor.is_add() && factor.to_string().starts_with('-'));
                    if negative {
                        format!(
                            "{} <- {} - ({})*{}",
                            target,
                            target,
                            normal(&crate::core::mul::neg(factor)?)?,
                            source
                        )
                    } else {
                        format!("{} <- {} + ({})*{}", target, target, factor, source)
                    }
                }
            }
        })
    }

    /// The elementary matrix of the operation (`matrix`).
    pub fn matrix(&self, n: usize) -> ExprMatrix {
        let e: ExprMatrix = ExprMatrix::eye(n);
        match self.kind {
            Kind::Swap => e
                .with_entry(self.target, self.target, int(0))
                .with_entry(self.source, self.source, int(0))
                .with_entry(self.target, self.source, int(1))
                .with_entry(self.source, self.target, int(1)),
            Kind::Scale => e.with_entry(self.target, self.target, self.factor.clone()),
            Kind::Add => e.with_entry(self.target, self.source, self.factor.clone()),
        }
    }

    /// The operation applied to a matrix (`apply`).
    pub fn apply(&self, m: &ExprMatrix) -> PolyResult<ExprMatrix> {
        self.matrix(m.rows).matmul(m)?.applyfunc(&normal)
    }
}

/// An operation with the matrix after it (`Step`).
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub operation: RowOperation,
    pub matrix: ExprMatrix,
}

/// The steps to a triangular form with the determinant (`DeterminantSteps`).
#[derive(Debug, Clone, PartialEq)]
pub struct DeterminantSteps {
    pub steps: Vec<Step>,
    pub exchanges: usize,
    pub diagonal: Vec<Basic>,
    pub value: Basic,
}

fn steps_with<F: Field>(field: &F, m: &ExprMatrix, reduced: bool, columns: usize) -> PolyResult<Vec<Step>> {
    let k: &F::D = field.domain();
    let (rows_n, cols_n) = m.shape();
    let mut rows: Vec<Vec<_>> = (0..rows_n)
        .map(|i| {
            m.row(i)
                .iter()
                .map(|e| field.element(e))
                .collect::<PolyResult<Vec<_>>>()
        })
        .collect::<PolyResult<_>>()?;
    let mut found: Vec<Step> = Vec::new();
    let record = |rows: &[Vec<<F::D as Domain>::Elem>],
                  found: &mut Vec<Step>,
                  kind: Kind,
                  target: usize,
                  source: usize,
                  factor: &<F::D as Domain>::Elem|
     -> PolyResult<()> {
        let mut entries: Vec<Basic> = Vec::new();
        for row in rows {
            for a in row {
                entries.push(field.expr(a)?);
            }
        }
        let matrix: ExprMatrix = ExprMatrix::new(rows_n, cols_n, entries)?;
        found.push(Step {
            operation: RowOperation {
                kind,
                target,
                source,
                factor: field.expr(factor)?,
            },
            matrix,
        });
        Ok(())
    };
    let mut r: usize = 0;
    for column in 0..cols_n.min(columns) {
        if r == rows_n {
            break;
        }
        let Some(chosen) = (r..rows_n).find(|i| !k.is_zero(&rows[*i][column])) else {
            continue;
        };
        if chosen != r {
            rows.swap(r, chosen);
            record(&rows, &mut found, Kind::Swap, r, chosen, &k.one())?;
        }
        let mut pivot = rows[r][column].clone();
        if reduced && !k.is_one(&pivot) {
            let inverse = k.invert(&pivot)?;
            rows[r] = rows[r].iter().map(|a| k.mul(a, &inverse)).collect();
            record(&rows, &mut found, Kind::Scale, r, r, &inverse)?;
            pivot = k.one();
        }
        let targets: Vec<usize> = if reduced {
            (0..rows_n).collect()
        } else {
            (r + 1..rows_n).collect()
        };
        for i in targets {
            if i == r || k.is_zero(&rows[i][column]) {
                continue;
            }
            let factor = k.neg(&k.exquo(&rows[i][column], &pivot)?);
            let pivot_row = rows[r].clone();
            rows[i] = rows[i]
                .iter()
                .zip(pivot_row.iter())
                .map(|(a, b)| {
                    if k.is_zero(b) {
                        a.clone()
                    } else {
                        k.add(a, &k.mul(&factor, b))
                    }
                })
                .collect();
            record(&rows, &mut found, Kind::Add, i, r, &factor)?;
        }
        r += 1;
    }
    Ok(found)
}

fn steps(m: &ExprMatrix, reduced: bool, columns: usize) -> PolyResult<Vec<Step>> {
    let kernels: Vec<Basic> = kernels_of(&[m]);
    if let Some(g) = complex_lift(&kernels)? {
        return steps_with(&GaussianEntries::new(g), m, reduced, columns);
    }
    if !kernels.is_empty() {
        let lift: PolynomialLift = PolynomialLift::new(&kernels);
        if !lift.exact {
            return Err(PolyError::NotImplemented(
                "elimination steps for a matrix whose entries are not rational functions of symbols, I and square roots of rationals"
                    .to_string(),
            ));
        }
        return steps_with(&FunctionEntries::new(lift), m, reduced, columns);
    }
    steps_with(&RationalEntries::new(), m, reduced, columns)
}

/// The steps of Gaussian elimination to an echelon form (`echelon_steps`).
pub fn echelon_steps(m: &ExprMatrix) -> PolyResult<Vec<Step>> {
    steps(m, false, m.cols)
}

/// The steps of Gauss–Jordan elimination (`rref_steps`).
pub fn rref_steps(m: &ExprMatrix) -> PolyResult<Vec<Step>> {
    steps(m, true, m.cols)
}

/// The steps reducing `[M | I]` to `[I | M**-1]` (`inverse_steps`).
pub fn inverse_steps(m: &ExprMatrix) -> PolyResult<Vec<Step>> {
    let n: usize = square(m)?;
    let augmented: ExprMatrix = m.col_insert(n, &ExprMatrix::eye(n))?;
    let found: Vec<Step> = steps(&augmented, true, n)?;
    let last: &ExprMatrix = found.last().map(|s| &s.matrix).unwrap_or(&augmented);
    for i in 0..n {
        for j in 0..n {
            if *last.at(i, j) != int(if i == j { 1 } else { 0 }) {
                return Err(value_error("the matrix is singular".to_string()));
            }
        }
    }
    Ok(found)
}

/// The steps reducing `[A | B]` (`solve_steps`).
pub fn solve_steps(a: &ExprMatrix, b: &ExprMatrix) -> PolyResult<Vec<Step>> {
    if a.rows != b.rows {
        return Err(value_error(format!(
            "{} equations and {} right-hand sides",
            a.rows, b.rows
        )));
    }
    let augmented: ExprMatrix = a.col_insert(a.cols, b)?;
    steps(&augmented, true, a.cols)
}

/// The determinant by elimination with its steps (`determinant_steps`).
pub fn determinant_steps(m: &ExprMatrix) -> PolyResult<DeterminantSteps> {
    let n: usize = square(m)?;
    let found: Vec<Step> = echelon_steps(m)?;
    let last: ExprMatrix = found.last().map(|s| s.matrix.clone()).unwrap_or_else(|| m.clone());
    let exchanges: usize = found.iter().filter(|s| s.operation.kind == Kind::Swap).count();
    let diagonal: Vec<Basic> = (0..n).map(|i| last.at(i, i).clone()).collect();
    let mut factors: Vec<Basic> = vec![int(if exchanges % 2 == 1 { -1 } else { 1 })];
    factors.extend(diagonal.iter().cloned());
    let value: Basic = normal(&crate::core::mul::mul(&factors)?)?;
    Ok(DeterminantSteps {
        steps: found,
        exchanges,
        diagonal,
        value,
    })
}

/// The matrix and the steps as text, the columns aligned (`format_steps`).
pub fn format_steps(m: &ExprMatrix, steps: &[Step]) -> PolyResult<String> {
    let render = |matrix: &ExprMatrix| -> Vec<String> {
        let cells: Vec<Vec<String>> = (0..matrix.rows)
            .map(|i| (0..matrix.cols).map(|j| matrix.at(i, j).to_string()).collect())
            .collect();
        let widths: Vec<usize> = (0..matrix.cols)
            .map(|j| (0..matrix.rows).map(|i| cells[i][j].chars().count()).max().unwrap_or(0))
            .collect();
        cells
            .iter()
            .map(|line| {
                let parts: Vec<String> = line
                    .iter()
                    .zip(widths.iter())
                    .map(|(c, w)| format!("{:>width$}", c, width = *w))
                    .collect();
                format!("[{}]", parts.join(", "))
            })
            .collect()
    };
    let mut lines: Vec<String> = render(m);
    for step in steps {
        lines.push(step.operation.repr()?);
        lines.extend(render(&step.matrix));
    }
    Ok(lines.join("\n"))
}
