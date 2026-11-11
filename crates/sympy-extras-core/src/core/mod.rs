//! Port of `sympy_extras.core`, the typed expression core: classes never
//! evaluate (`Add(x, x)` stays), lowercase functions do (`add`, `mul`,
//! `pow`, ...).
//!
//! # The representation
//!
//! Python's core is an open class hierarchy whose nodes are a class and a
//! tuple of arguments, equal when both are equal. The port keeps exactly
//! that shape with one node type:
//!
//! * [`Basic`] is a shared immutable node, `Arc<Node>` with the node's
//!   [`Head`] (its class, and the data of an atom: the value of a number,
//!   the name and assumptions of a symbol) and its arguments; the hash and
//!   the `compare` key are computed once, as Python caches them;
//! * [`Head`] has one variant per class of the core; the function classes,
//!   an open set in Python (`sin`, `gamma`, user functions), will be one
//!   variant holding a shared function class;
//! * `e.func(*e.args) == e` is [`Basic::rebuild`]; equality and hashing are
//!   structural, with the special cases of Python (a `Rational` equals the
//!   `Integer` of the same value; a `Dummy` only equals itself).
//!
//! The modules are ported bottom-up: `assumptions` (the rules and their
//! closure), `sorting` (the `compare` order), `basic`, `numbers` (with the
//! arithmetic of the numbers), `parameters`, `symbol` so far; the evaluating functions, printing and the rest follow.

// a node caches its `compare` key in a `OnceLock`; the hash and the equality
// never read it, so nodes are sound keys of maps and sets
#![allow(clippy::mutable_key_type)]

pub mod add;
pub mod assumptions;
pub mod basic;
pub mod evalf;
pub mod expr;
pub mod mul;
pub mod numbers;
pub mod parameters;
pub mod power;
pub mod printing;
pub mod sorting;
pub mod symbol;

pub use basic::{Basic, Function, Head, Node};
pub use symbol::{Dummy, Symbol, Wild};
