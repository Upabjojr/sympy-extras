//! Port of `sympy_extras/diffgeom/tensors.py`: tensors of any rank on a
//! metric, the components in row-major order of the indices, each index
//! up (`u`) or down (`d`): sums, products, transpositions,
//! (anti)symmetrisation, raising and lowering, contraction, trace,
//! covariant derivative and divergence.

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;

use crate::core::basic::Basic;
use crate::error::{value_error, PolyError, PolyResult};
use crate::matrices::dense::ExprMatrix;

use super::metric::Metric;

type R = PolyResult<Basic>;
type Index = Vec<usize>;

fn int(n: i64) -> Basic {
    Basic::int(n)
}
fn zero() -> Basic {
    int(0)
}
fn m2(a: &Basic, b: &Basic) -> R {
    crate::core::mul::mul(&[a.clone(), b.clone()])
}

fn product(n: usize, rank: usize) -> Vec<Index> {
    let mut out: Vec<Index> = vec![Vec::new()];
    for _ in 0..rank {
        let mut next: Vec<Index> = Vec::new();
        for prefix in &out {
            for i in 0..n {
                let mut v: Index = prefix.clone();
                v.push(i);
                next.push(v);
            }
        }
        out = next;
    }
    out
}

/// A tensor on a metric (`Tensor`).
#[derive(Debug, Clone)]
pub struct Tensor {
    pub metric: Arc<Metric>,
    pub components: Vec<Basic>,
    pub positions: String,
    pub rank: usize,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.metric, &other.metric)
            && self.positions == other.positions
            && self.components == other.components
    }
}

impl Tensor {
    pub fn new(metric: &Arc<Metric>, components: Vec<Basic>, positions: &str) -> PolyResult<Tensor> {
        if positions.chars().any(|p| p != 'u' && p != 'd') {
            return Err(value_error(format!(
                "the positions of the indices are 'u' or 'd': '{}'",
                positions
            )));
        }
        let rank: usize = positions.chars().count();
        let n: usize = metric.dimension;
        if components.len() != n.pow(rank as u32) {
            return Err(value_error(format!(
                "{} components for a tensor of rank {} in dimension {}",
                components.len(),
                rank,
                n
            )));
        }
        Ok(Tensor {
            metric: metric.clone(),
            components,
            positions: positions.to_string(),
            rank,
        })
    }

    fn position(&self, k: usize) -> char {
        self.positions.as_bytes()[k] as char
    }

    fn offset(&self, indices: &[usize]) -> PolyResult<usize> {
        let n: usize = self.metric.dimension;
        if indices.len() != self.rank || indices.iter().any(|i| *i >= n) {
            let shown: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
            let tuple: String = if shown.len() == 1 {
                format!("({},)", shown[0])
            } else {
                format!("({})", shown.join(", "))
            };
            return Err(PolyError::Index(format!(
                "the indices {} of a tensor of rank {} in dimension {}",
                tuple, self.rank, n
            )));
        }
        Ok(indices.iter().fold(0, |offset, i| offset * n + i))
    }

    /// The component at the indices (`__getitem__`).
    pub fn get(&self, indices: &[usize]) -> R {
        Ok(self.components[self.offset(indices)?].clone())
    }

    fn indices(&self) -> Vec<Index> {
        product(self.metric.dimension, self.rank)
    }

    fn build(&self, positions: &str, values: &[(Index, Basic)]) -> PolyResult<Tensor> {
        let n: usize = self.metric.dimension;
        let rank: usize = positions.chars().count();
        let mut components: Vec<Basic> = Vec::new();
        for index in product(n, rank) {
            components.push(match values.iter().find(|(k, _)| *k == index) {
                Some((_, v)) => (self.metric.simplifier)(v)?,
                None => zero(),
            });
        }
        Tensor::new(&self.metric, components, positions)
    }

    pub fn is_zero(&self) -> PolyResult<bool> {
        for c in &self.components {
            if (self.metric.simplifier)(c)? != zero() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn repr(&self) -> String {
        let shown: Vec<String> = self.components.iter().map(|c| c.to_string()).collect();
        format!("Tensor('{}', [{}])", self.positions, shown.join(", "))
    }

    fn same_kind(&self, other: &Tensor) -> PolyResult<()> {
        if !Arc::ptr_eq(&self.metric, &other.metric) || other.positions != self.positions {
            return Err(value_error(format!(
                "tensors of different kinds: '{}' and '{}'",
                self.positions, other.positions
            )));
        }
        Ok(())
    }

    fn zip_with(&self, other: &Tensor, f: fn(&Basic, &Basic) -> R) -> PolyResult<Tensor> {
        self.same_kind(other)?;
        let components: Vec<Basic> = self
            .components
            .iter()
            .zip(other.components.iter())
            .map(|(a, b)| (self.metric.simplifier)(&f(a, b)?))
            .collect::<PolyResult<_>>()?;
        Tensor::new(&self.metric, components, &self.positions)
    }

    pub fn add(&self, other: &Tensor) -> PolyResult<Tensor> {
        self.zip_with(other, |a, b| crate::core::add::add(&[a.clone(), b.clone()]))
    }

    pub fn sub(&self, other: &Tensor) -> PolyResult<Tensor> {
        self.zip_with(other, crate::core::add::sub)
    }

    pub fn neg(&self) -> PolyResult<Tensor> {
        let components: Vec<Basic> = self
            .components
            .iter()
            .map(crate::core::mul::neg)
            .collect::<PolyResult<_>>()?;
        Tensor::new(&self.metric, components, &self.positions)
    }

    pub fn scale(&self, factor: &Basic) -> PolyResult<Tensor> {
        let components: Vec<Basic> = self
            .components
            .iter()
            .map(|a| (self.metric.simplifier)(&m2(factor, a)?))
            .collect::<PolyResult<_>>()?;
        Tensor::new(&self.metric, components, &self.positions)
    }

    /// The tensor product (`__mul__`).
    pub fn mul(&self, other: &Tensor) -> PolyResult<Tensor> {
        if !Arc::ptr_eq(&self.metric, &other.metric) {
            return Err(value_error("tensors on different metrics".to_string()));
        }
        let mut components: Vec<Basic> = Vec::new();
        for a in &self.components {
            for b in &other.components {
                components.push(if *a != zero() && *b != zero() {
                    (self.metric.simplifier)(&m2(a, b)?)?
                } else {
                    zero()
                });
            }
        }
        Tensor::new(
            &self.metric,
            components,
            &format!("{}{}", self.positions, other.positions),
        )
    }

    pub fn transpose(&self, i: usize, j: usize) -> PolyResult<Tensor> {
        let mut positions: Vec<char> = self.positions.chars().collect();
        positions.swap(i, j);
        let mut values: Vec<(Index, Basic)> = Vec::new();
        for index in self.indices() {
            let mut swapped: Index = index.clone();
            swapped.swap(i, j);
            values.push((index, self.get(&swapped)?));
        }
        self.build(&positions.into_iter().collect::<String>(), &values)
    }

    pub fn symmetrize(&self, i: usize, j: usize) -> PolyResult<Tensor> {
        if self.position(i) != self.position(j) {
            return Err(value_error(format!(
                "the indices {} and {} are in different positions",
                i, j
            )));
        }
        self.add(&self.transpose(i, j)?)?
            .scale(&crate::core::mul::div(&int(1), &int(2))?)
    }

    pub fn antisymmetrize(&self, i: usize, j: usize) -> PolyResult<Tensor> {
        if self.position(i) != self.position(j) {
            return Err(value_error(format!(
                "the indices {} and {} are in different positions",
                i, j
            )));
        }
        self.sub(&self.transpose(i, j)?)?
            .scale(&crate::core::mul::div(&int(1), &int(2))?)
    }

    fn moved(&self, k: usize, up: bool) -> PolyResult<Tensor> {
        let n: usize = self.metric.dimension;
        if k >= self.rank {
            return Err(PolyError::Index(format!(
                "no index {} in a tensor of rank {}",
                k, self.rank
            )));
        }
        if (self.position(k) == 'u') == up {
            return Ok(self.clone());
        }
        let matrix: ExprMatrix = if up {
            self.metric.inverse()?
        } else {
            self.metric.g.clone()
        };
        let mut values: Vec<(Index, Basic)> = Vec::new();
        for index in self.indices() {
            let mut terms: Vec<Basic> = Vec::new();
            for m in 0..n {
                let factor: &Basic = matrix.at(index[k], m);
                if *factor == zero() {
                    continue;
                }
                let mut source: Index = index.clone();
                source[k] = m;
                let component: Basic = self.get(&source)?;
                if component != zero() {
                    terms.push(m2(factor, &component)?);
                }
            }
            if !terms.is_empty() {
                values.push((index, crate::core::add::add(&terms)?));
            }
        }
        let mut positions: Vec<char> = self.positions.chars().collect();
        positions[k] = if up { 'u' } else { 'd' };
        self.build(&positions.into_iter().collect::<String>(), &values)
    }

    pub fn raise_index(&self, k: usize) -> PolyResult<Tensor> {
        self.moved(k, true)
    }

    pub fn lower_index(&self, k: usize) -> PolyResult<Tensor> {
        self.moved(k, false)
    }

    pub fn with_positions(&self, positions: &str) -> PolyResult<Tensor> {
        if positions.chars().count() != self.rank {
            return Err(value_error(format!(
                "{} positions for a tensor of rank {}",
                positions.chars().count(),
                self.rank
            )));
        }
        let mut result: Tensor = self.clone();
        for (k, p) in positions.chars().enumerate() {
            result = result.moved(k, p == 'u')?;
        }
        Ok(result)
    }

    pub fn contract(&self, i: usize, j: usize) -> PolyResult<Tensor> {
        let pair: [char; 2] = if i < self.rank && j < self.rank {
            [self.position(i), self.position(j)]
        } else {
            ['?', '?']
        };
        if i == j || !(pair.contains(&'u') && pair.contains(&'d')) {
            return Err(value_error(
                "a contraction takes one upper and one lower index".to_string(),
            ));
        }
        let n: usize = self.metric.dimension;
        let kept: Vec<usize> = (0..self.rank).filter(|k| *k != i && *k != j).collect();
        let mut values: Vec<(Index, Basic)> = Vec::new();
        for index in product(n, kept.len()) {
            let mut terms: Vec<Basic> = Vec::new();
            for m in 0..n {
                let mut full: Index = vec![0; self.rank];
                for (position, value) in kept.iter().zip(index.iter()) {
                    full[*position] = *value;
                }
                full[i] = m;
                full[j] = m;
                let component: Basic = self.get(&full)?;
                if component != zero() {
                    terms.push(component);
                }
            }
            if !terms.is_empty() {
                values.push((index, crate::core::add::add(&terms)?));
            }
        }
        let positions: String = kept.iter().map(|k| self.position(*k)).collect();
        self.build(&positions, &values)
    }

    pub fn trace(&self) -> R {
        if self.rank != 2 {
            return Err(value_error("the trace of a tensor of rank two".to_string()));
        }
        let mixed: Tensor = match self.positions.as_str() {
            "ud" | "du" => self.clone(),
            "dd" => self.raise_index(0)?,
            _ => self.lower_index(0)?,
        };
        Ok(mixed.contract(0, 1)?.components[0].clone())
    }

    pub fn scalar(&self) -> R {
        if self.rank != 0 {
            return Err(value_error(format!("not a scalar: rank {}", self.rank)));
        }
        Ok(self.components[0].clone())
    }

    pub fn covariant_derivative(&self) -> PolyResult<Tensor> {
        let n: usize = self.metric.dimension;
        let x: &[Basic] = &self.metric.coordinates;
        let mut values: Vec<(Index, Basic)> = Vec::new();
        for index in self.indices() {
            for c in 0..n {
                let mut terms: Vec<Basic> = vec![crate::core::function::derivative(&self.get(&index)?, &x[c])?];
                for (k, position) in self.positions.chars().enumerate() {
                    for m in 0..n {
                        let mut source: Index = index.clone();
                        source[k] = m;
                        let component: Basic = self.get(&source)?;
                        if component == zero() {
                            continue;
                        }
                        if position == 'u' {
                            let gamma: Basic = self.metric.christoffel(index[k], c, m)?;
                            if gamma != zero() {
                                terms.push(m2(&gamma, &component)?);
                            }
                        } else {
                            let gamma: Basic = self.metric.christoffel(m, c, index[k])?;
                            if gamma != zero() {
                                terms.push(m2(&crate::core::mul::neg(&gamma)?, &component)?);
                            }
                        }
                    }
                }
                let mut key: Index = index.clone();
                key.push(c);
                values.push((key, crate::core::add::add(&terms)?));
            }
        }
        self.build(&format!("{}d", self.positions), &values)
    }

    pub fn divergence(&self, k: usize) -> PolyResult<Tensor> {
        self.raise_index(k)?.covariant_derivative()?.contract(k, self.rank)
    }
}

pub fn metric_tensor(metric: &Arc<Metric>) -> PolyResult<Tensor> {
    Tensor::new(metric, metric.g.entries.clone(), "dd")
}

pub fn riemann_tensor(metric: &Arc<Metric>) -> PolyResult<Tensor> {
    let n: usize = metric.dimension;
    let mut components: Vec<Basic> = Vec::new();
    for index in product(n, 4) {
        components.push(metric.riemann(index[0], index[1], index[2], index[3])?);
    }
    Tensor::new(metric, components, "uddd")
}

pub fn ricci_tensor(metric: &Arc<Metric>) -> PolyResult<Tensor> {
    Tensor::new(metric, metric.ricci()?.entries, "dd")
}

pub fn einstein_tensor(metric: &Arc<Metric>) -> PolyResult<Tensor> {
    Tensor::new(metric, metric.einstein()?.entries, "dd")
}
