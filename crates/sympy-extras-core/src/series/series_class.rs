//! The base of the series, ported from `sympy_extras.series.series_class`
//! (SymPy's `SeriesBase`): an expression whose terms are indexed by the
//! integers of an interval `start, start + 1, ..., stop`, with `term`, the
//! iteration over the terms and their slicing. The Python class is a trait
//! here: an implementer gives `start`, `stop`, `length` and `eval_term`.

use crate::core::assumptions::Predicate;
use crate::core::basic::{Basic, Head};
use crate::core::expr::is_true;
use crate::error::{value_error, PolyError, PolyResult};

/// `(b - a).is_extended_positive is True` (`_less`).
fn less(a: &Basic, b: &Basic) -> PolyResult<bool> {
    Ok(is_true(&crate::core::add::sub(b, a)?, Predicate::extended_positive))
}

/// A series (`SeriesBase`).
pub trait SeriesBase {
    /// The first point of the series (included).
    fn start(&self) -> PolyResult<Basic>;

    /// The last point of the series (included).
    fn stop(&self) -> PolyResult<Basic>;

    /// The number of terms (`oo` for an infinite series).
    fn length(&self) -> PolyResult<Basic>;

    /// The term at a point in the range (`_eval_term`).
    fn eval_term(&self, pt: &Basic) -> PolyResult<Basic>;

    /// The bound variables.
    fn variables(&self) -> Vec<Basic> {
        Vec::new()
    }

    /// The term at the point `pt`; `IndexError` outside the range.
    fn term(&self, pt: &Basic) -> PolyResult<Basic> {
        let (start, stop) = (self.start()?, self.stop()?);
        if less(pt, &start)? || less(&stop, pt)? {
            return Err(PolyError::Index(format!(
                "Index {} out of bounds ({}, {})",
                pt, start, stop
            )));
        }
        self.eval_term(pt)
    }

    /// The `i`-th point: from the start, or back from the stop when the
    /// series starts at `-oo` (`_ith_point`).
    fn ith_point(&self, i: i64) -> PolyResult<Basic> {
        let start: Basic = self.start()?;
        let offset: Basic = Basic::int(i);
        if matches!(start.head(), Head::NegativeInfinity) {
            return crate::core::add::sub(&self.stop()?, &offset);
        }
        crate::core::add::add(&[start, offset])
    }

    /// The number of terms as a bound, `None` for an infinite series
    /// (`_length_bound`).
    fn length_bound(&self) -> PolyResult<Option<i64>> {
        let length: Basic = self.length()?;
        if matches!(length.head(), Head::Infinity) {
            return Ok(None);
        }
        if let (
            Head::Rational {
                integer_class: true, ..
            },
            Some(v),
        ) = (length.head(), length.as_rational())
        {
            return i64::try_from(v.numerator())
                .map(Some)
                .map_err(|_| value_error(format!("the length {} is too large", length)));
        }
        Err(value_error(format!("the length of the series {} is not known", length)))
    }

    /// The `index`-th term (`__getitem__` with an integer).
    fn item(&self, index: i64) -> PolyResult<Basic> {
        self.term(&self.ith_point(index)?)
    }

    /// The terms of a slice `start:stop:step` (`__getitem__` with a slice;
    /// the stop may be left out for a finite series).
    fn slice(&self, start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> PolyResult<Vec<Basic>> {
        let start: i64 = start.unwrap_or(0);
        let stop: i64 = match stop {
            Some(s) => s,
            None => self
                .length_bound()?
                .ok_or_else(|| value_error("an infinite series needs the end of the slice".to_string()))?,
        };
        let step: i64 = step.unwrap_or(1);
        if step == 0 {
            return Err(value_error("slice step cannot be zero".to_string()));
        }
        let mut terms: Vec<Basic> = Vec::new();
        let mut i: i64 = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            terms.push(self.item(i)?);
            i += step;
        }
        Ok(terms)
    }

    /// The first `count` terms, at most the length (`__iter__`).
    fn first_terms(&self, count: usize) -> PolyResult<Vec<Basic>> {
        let bound: Option<i64> = self.length_bound()?;
        let mut terms: Vec<Basic> = Vec::new();
        let mut i: i64 = 0;
        while (terms.len() < count) && bound.map_or(true, |b| i < b) {
            terms.push(self.item(i)?);
            i += 1;
        }
        Ok(terms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The squares `k**2` for `k` from `start` to `stop`.
    struct Squares {
        start: Basic,
        stop: Basic,
    }

    impl SeriesBase for Squares {
        fn start(&self) -> PolyResult<Basic> {
            Ok(self.start.clone())
        }

        fn stop(&self) -> PolyResult<Basic> {
            Ok(self.stop.clone())
        }

        fn length(&self) -> PolyResult<Basic> {
            if matches!(self.stop.head(), Head::Infinity) || matches!(self.start.head(), Head::NegativeInfinity) {
                return Ok(Basic::oo());
            }
            crate::core::add::add(&[crate::core::add::sub(&self.stop, &self.start)?, Basic::one()])
        }

        fn eval_term(&self, pt: &Basic) -> PolyResult<Basic> {
            crate::core::power::pow(pt, &Basic::int(2))
        }
    }

    fn shown(v: &[Basic]) -> Vec<String> {
        v.iter().map(|e| e.to_string()).collect()
    }

    #[test]
    fn finite_and_infinite() {
        let s = Squares {
            start: Basic::int(-2),
            stop: Basic::int(3),
        };
        assert_eq!(
            shown(&s.slice(None, None, None).unwrap()),
            ["4", "1", "0", "1", "4", "9"]
        );
        assert_eq!(shown(&s.slice(Some(1), Some(5), Some(2)).unwrap()), ["1", "1"]);
        assert_eq!(s.item(5).unwrap().to_string(), "9");
        assert!(matches!(s.item(6), Err(PolyError::Index(_))));
        assert!(matches!(s.term(&Basic::int(-3)), Err(PolyError::Index(_))));
        let back = Squares {
            start: Basic::negative_oo(),
            stop: Basic::int(0),
        };
        assert_eq!(shown(&back.first_terms(3).unwrap()), ["0", "1", "4"]);
        assert!(back.slice(None, None, None).is_err());
        let up = Squares {
            start: Basic::int(1),
            stop: Basic::oo(),
        };
        assert_eq!(shown(&up.slice(None, Some(3), None).unwrap()), ["1", "4", "9"]);
    }
}
