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

## Conventions

See the crate documentation of `sympy-extras-core` (`src/lib.rs`): Python
`int` is `numbers::Int`, `Fraction` is `numbers::Rat` (both from `dashu`, whose
gcd is faster than CPython's; `num-bigint`'s was 12 times slower at 50 000 bits),
degrees and indices are `usize`, data-dependent exceptions are
`Err(PolyError)`, `NotImplementedError` is a panic.

## Status

| Python module | Rust | Notes |
|---|---|---|
| `polys/domains/domain.py` | done | trait `Domain`; `to_expr`/`from_expr`, `factor_dense` and the root isolation hooks wait for the ported modules |
| `polys/domains/integerring.py` | done | |
| `polys/domains/rationalfield.py` | done | with the fraction-free sparse echelon form |
| `polys/domains/finitefield.py` | done | elements `u64`, products in `u128` (any modulus below `2**64`) |
| `polys/densebasic.py` | done | |
| `polys/densearith.py` | done | |
| `polys/densetools.py` | done | |
| `polys/euclidtools.py` | done | |
| `polys/sqfreetools.py` | done | |
| everything else | not yet | next: `galoistools`, `factortools`, `heuristicgcd`, the polynomial ring domains, `rings` |
