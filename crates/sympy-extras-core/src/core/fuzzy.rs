//! Three-valued logic on `Option<bool>`, ported from
//! `sympy_extras.core.fuzzy`: `fuzzy_and`, `fuzzy_or`, `fuzzy_not` (the
//! implementations shared with `sets`) and [`fuzzy_group`].

// PORTED FROM SYMPY: sympy/core/logic.py (fuzzy_and, fuzzy_or, fuzzy_not,
// _fuzzy_group) (SymPy 1.14)
// Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
// see sympy_extras/LICENSE-SymPy.

pub use crate::sets::sets::{fuzzy_and, fuzzy_not, fuzzy_or};

// PORTED FROM SYMPY: sympy/core/logic.py:_fuzzy_group
/// `Some(true)` when all are true, `None` when one is unknown, else
/// `Some(false)` (`None` at a second false with `quick_exit`).
pub fn fuzzy_group(args: impl IntoIterator<Item = Option<bool>>, quick_exit: bool) -> Option<bool> {
    let mut saw_other: bool = false;
    for a in args {
        match a {
            Some(true) => continue,
            None => return None,
            Some(false) => {
                if quick_exit && saw_other {
                    return None;
                }
                saw_other = true;
            }
        }
    }
    Some(!saw_other)
}
