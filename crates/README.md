# The Rust port of sympy-extras

- `sympy-extras-core`: pure Rust, no Python dependency, for crates.io. Its
  modules mirror `sympy_extras` one to one (`sympy_extras/polys/densearith.py`
  is `src/polys/densearith.rs`, with the same function names).
- `sympy-extras-py`: PyO3 bindings, distributed on PyPI as `sympy-extras-rs`
  (module `sympy_extras_rs`), separately from the pure Python `sympy-extras`.
  The submodules have the names and signatures of the Python ones, so the
  tests in `../tests` run against both.

## Building and testing

```
cargo test                                   # the Rust unit tests
cd crates/sympy-extras-py && maturin develop --release && cd ../..
python -m pytest tests                       # both implementations
```

The workspace members are listed explicitly in `Cargo.toml` (the session
directory `crates/.claude/` must not be picked up as a crate).

```
```

## Conventions

See the crate documentation of `sympy-extras-core` (`src/lib.rs`): Python
`int` is `numbers::Int`, `Fraction` is `numbers::Rat` (both from `dashu`, whose
gcd is faster than CPython's; `num-bigint`'s was 12 times slower at 50 000 bits),
degrees and indices are `usize`, data-dependent exceptions are
`Err(PolyError)`, `NotImplementedError` is a panic.

## Status

| Python module | Rust | Notes |
|---|---|---|
| `polys/domains/domain.py` | done | trait `Domain` with `factor_dense`; `to_expr`/`from_expr` and the root isolation hooks wait for the ported modules |
| `polys/domains/integerring.py` | done | |
| `polys/domains/rationalfield.py` | done | with the fraction-free sparse echelon form |
| `polys/domains/finitefield.py` | done | elements `u64`, products in `u128` (any modulus below `2**64`) |
| `polys/densebasic.py` | done | |
| `polys/densearith.py` | done | |
| `polys/densetools.py` | done | |
| `polys/euclidtools.py` | done | |
| `polys/sqfreetools.py` | done | |
| `polys/galoistools.py` | done | the random choices from a seeded SplitMix64 (`random.rs`); the factorisations are sorted, so equal to Python's |
| `polys/factortools.py` | over `ZZ`, `QQ` | the number-field part (`dup_ext_*`, Trager) needs `AlgebraicField` |
| `polys/rings.py` | done | orders as the enum `MonomialOrder` (lex, grlex, grevlex, and the weight, block and homogeneous orders of `orderings.py` and `modulargroebner.py`); `from_expr`/`to_expr` wait for the core |
| `polys/heuristicgcd.py` | done | |
| `polys/polyerrors.py` | `HeuristicGCDFailed` | the others as they are needed |
| `polys/domains/sparsepolynomialring.py` | done | the Integer/Rational flavours as `SparsePolynomialRing::sympy` with `SparseGround` hooks; `IteratedPolynomialRing` one recursive type; over `ZZ`/`QQ` (the gcd needs an ordered ground, in Python too) |
| `polys/domains/polynomialring.py` | done | the tower depth is a type (`PolynomialRing<PolynomialRing<K>>`); the bindings offer `K[y]` over `ZZ`, `QQ`, `GF(p)`, on which every dense binding runs |
| `polys/domains/fractionfield.py` | done | conversions between different fields of fractions wait for the core (they go through expressions) |
| everything else | not yet | next: `polyclasses`, `f4`, `subresultants`, `rootisolation` |
