//! `sympify`, ported from `sympy_extras.core.sympify`: the conversion of
//! the values a caller may give (nodes, integers, fractions, booleans,
//! strings, sequences) to the expression tree. Python's dynamic argument
//! is the typed [`Sympifiable`]; floats need `Float`, which the Rust core
//! does not have, and the dictionaries a `Dict` node, which it does not
//! have either, and the foreign objects of the bridge do not exist here.

use crate::core::basic::{Basic, Head};
use crate::error::{PolyError, PolyResult};
use crate::numbers::{Int, Rat};

/// What `sympify` accepts.
#[derive(Debug, Clone)]
pub enum Sympifiable {
    Basic(Basic),
    Integer(Int),
    Fraction(Rat),
    Bool(bool),
    Str(String),
    Sequence(Vec<Sympifiable>),
}

/// `SympifyError`, a `ValueError`.
fn sympify_error(message: String) -> PolyError {
    PolyError::Value(format!("SympifyError: {}", message))
}

/// `a` as a node of the expression tree (`sympify`).
pub fn sympify(a: &Sympifiable) -> PolyResult<Basic> {
    match a {
        Sympifiable::Basic(b) => Ok(b.clone()),
        Sympifiable::Bool(v) => Ok(crate::core::logic::boolean(*v)),
        Sympifiable::Integer(n) => Ok(Basic::integer(n.clone())),
        Sympifiable::Fraction(q) => Ok(Basic::from_rational(q.clone())),
        Sympifiable::Str(text) => match crate::core::parsing::parse(text, crate::core::parameters::evaluate()) {
            Ok(e) => Ok(e),
            // outside the native grammar: no foreign parser in the Rust core
            Err(PolyError::Value(message)) if message.starts_with("ParseUnsupported") => {
                Err(sympify_error(format!("cannot parse {:?}", text)))
            }
            Err(other) => Err(other),
        },
        Sympifiable::Sequence(items) => Ok(Basic::new(
            Head::Tuple,
            items.iter().map(sympify).collect::<PolyResult<Vec<Basic>>>()?,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions() {
        let s = |a: Sympifiable| sympify(&a).map(|b| b.to_string());
        assert_eq!(s(Sympifiable::Integer(Int::from(2))).unwrap(), "2");
        assert_eq!(s(Sympifiable::Fraction(crate::numbers::rat(1, 2))).unwrap(), "1/2");
        assert_eq!(s(Sympifiable::Bool(true)).unwrap(), "True");
        assert_eq!(s(Sympifiable::Str("x**2 + 1".to_string())).unwrap(), "x**2 + 1");
        assert_eq!(
            s(Sympifiable::Sequence(vec![
                Sympifiable::Integer(Int::from(1)),
                Sympifiable::Str("y".to_string())
            ]))
            .unwrap(),
            "(1, y)"
        );
        assert!(
            matches!(s(Sympifiable::Str("x @ y".to_string())), Err(PolyError::Value(m)) if m.starts_with("SympifyError"))
        );
    }
}
