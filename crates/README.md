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

## Conventions

See the crate documentation of `sympy-extras-core` (`src/lib.rs`): Python
`int` is `numbers::Int`, `Fraction` is `numbers::Rat` (both from `dashu`, whose
gcd is faster than CPython's; `num-bigint`'s was 12 times slower at 50 000 bits),
degrees and indices are `usize` (the characteristic of a domain is an `Int`), data-dependent exceptions are
`Err(PolyError)`, `NotImplementedError` is a panic.

## Status

| Python module | Rust | Notes |
|---|---|---|
| `polys/domains/domain.py` | done | trait `Domain` with `factor_dense` and `nullspace` (`try_sparse_echelon` gives the `ValueError` of a non-unit pivot over a ring); `to_expr`/`from_expr` and the root isolation hooks wait for the ported modules |
| `polys/domains/integerring.py` | done | |
| `polys/domains/rationalfield.py` | done | with the fraction-free sparse echelon form |
| `polys/domains/algebraicfield.py` | done | elements `Vec<Rat>`; `alpha_expr` a core node (the bindings take any object and name the root by its `str`); `to_expr`/`from_expr` and the conversion between different fields wait for the core; a separate Python class in the bindings, not a `DomainKind` |
| `polys/domains/gaussiandomains.py` | done | `ZZ_I`, `QQ_I` with the elements `GaussianInteger`/`GaussianRational`; kinds of `Domain` in the bindings, so every dense binding runs over them; `to_expr`/`from_expr` wait for the core |
| `polys/domains/finitefield.py` | done | elements `u64`, products in `u128` (any modulus below `2**64`) |
| `polys/densebasic.py` | done | |
| `polys/densearith.py` | done | |
| `polys/densetools.py` | done | |
| `polys/euclidtools.py` | done | |
| `polys/sqfreetools.py` | done | |
| `polys/galoistools.py` | done | the random choices from a seeded SplitMix64 (`random.rs`); the factorisations are sorted, so equal to Python's |
| `polys/factortools.py` | done | with Trager's algorithm over number fields (`dup_ext_norm`, `dup_ext_sqf_norm`, `dup_ext_factor`) and van Hoeij's knapsack recombination (`dup_zz_van_hoeij`, from ten modular factors on) |
| `polys/lattice.py` | done | de Weger's integral LLL, Hermite and Smith normal forms, integer kernels and systems; `ArithmeticError` as `PolyError::Arithmetic`; 2-4x faster than Python |
| `polys/rings.py` | done | orders as the enum `MonomialOrder` (lex, grlex, grevlex, and the weight, block and homogeneous orders, now in `rings.py` too); `as_order`, `WeightOrder`, `BlockOrder`, `elimination_order` in the bindings (the two classes as functions returning a `MonomialOrder`); `MonomialOrder::is_graded` is the test of `f4.is_graded` (nothing reads the `is_graded` attribute of the Python orders); `from_expr`/`to_expr` wait for the core |
| `polys/heuristicgcd.py` | done | |
| `polys/polyerrors.py` | `HeuristicGCDFailed` | the others as they are needed |
| `polys/domains/sparsepolynomialring.py` | done | the Integer/Rational flavours as `SparsePolynomialRing::sympy` with `SparseGround` hooks; `IteratedPolynomialRing` one recursive type; over `ZZ`/`QQ` (the gcd needs an ordered ground, in Python too) |
| `polys/domains/polynomialring.py` | done | the tower depth is a type (`PolynomialRing<PolynomialRing<K>>`); the bindings offer `K[y]` over `ZZ`, `QQ`, `GF(p)`, on which every dense binding runs |
| `polys/domains/fractionfield.py` | done | conversions between different fields of fractions wait for the core (they go through expressions) |
| `polys/domains/sparsefractionfield.py` | done | over the sparse rings of `ZZ` and `QQ` (with `cancel`, and `RationalPolynomialRing.cancel` as a `SparseGround` hook); `to_expr`/`from_expr` wait for the core |
| `polys/rootisolation.py` | done | with the `isolate_real_roots_dense`/`count_real_roots_dense` hooks of `ZZ` and `QQ` |
| `polys/f4.py` | done | Katsura-4: the same bases, 2.5x (lex over `QQ`) to 7x (`GF(32003)`) faster than Python, untuned |
| `polys/stronggroebner.py` | done | 10-20x faster than Python on the random test systems |
| `polys/subresultants.py` | `dup_psc` | `dmp_psc` is `dup_psc` over a `PolynomialRing` tower; `psc` on SymPy's `PolyElement` has no counterpart |
| `polys/domains/galoisfield.py` | done | `GaloisField` (a word-sized prime, as `FiniteField`), its own Python class in the bindings (`DomainKind` is `Copy` and cannot hold a modulus); `to_str` prints the generator by name, `to_expr` waits for the core |
| `polys/galoisfactor.py` | done | the equal-degree stage on the seeded `Rng`; the factors are sorted, so the result is Python's |
| `polys/complexroots.py` | done | `Rectangle` a struct (a class in the bindings, compared by its corners); the factors of `dup_isolate_complex_roots` told apart by index instead of list identity |
| `polys/numberfields/basis.py` | done | the Round 2 algorithm; the multiplication table computed with the order (Python: on first use); `minpoly.py` waits for the core |
| `polys/numberfields/qqbar.py` | done | the isolating region behind a mutex (refined in place as in Python, the methods take `&self`); `minpoly._sum_resultant`/`_product_resultant` as `sum_resultant`/`product_resultant`; Python's `%.10g` for the reprs; the `TypeError`s of the decisions in the bindings |
| `polys/numberfields/recognize.py` | done | |
| `polys/zerodim.py` | done | the pivot of `univariate_in_ideal` is the largest monomial (the dependence does not depend on it); over the `QQ` rings of the bindings |
| `polys/algebraicfactor.py` | done | `AlgebraicPolynomialRing` its own `Domain` (no order, monic gcd, recursive multivariate gcd); `PolyRing` over an `AlgebraicField` in the bindings (`RingKind::Algebraic`) |
| `simplify/signsimp.py` | done | `signsimp`, `sub_pre`, `sub_post`; `replace_where` (`replace` with a predicate) and `xreplace_evaluated` in `core/function.rs` |
| `polys/polytools.py` | partly | `_factor_all` (`factor_all`), `_resultant_first`, and the class `Poly` on the core: `sort_gens`, the decomposition with and without generators, the domains `ZZ`, `QQ`, `ZZ[a]`, `QQ[a]` (`PolyDomain`, `construct_domain`), printing, terms, degrees, arithmetic, `gcd`, `lcm`, `cofactors`, `cancel`, `div`, `exquo`, `monic`, `content`, `primitive`, `eval`, `resultant`, `discriminant`, `sqf_list`, `factor_list` (sorted by `_domain_key` over a polynomial domain); the errors of `polyerrors` are `PolyError::Polys`; the functions on expressions `poly`, `degree`, `LC`, `gcd`, `lcm`, `gcd_list`, `lcm_list`, `resultant`, `discriminant`, `invert`, `cancel_fraction`, `cancel` (with and without generators; `ZZ_I`, `RR`, `EX` and the SymPy fallbacks `NotImplemented`); not yet: `factor`, `factor_list`, `sqf`, `sqf_list` of the module |
| `polys/regularchains/recursive.py`, `operations.py` | done | the pool of factors in insertion order, chains told apart by position where Python compares identities, caches keyed by terms; `Decomposer` a mutable class in the bindings; `regularchain.py` waits for the core |
| `polys/fglm.py` | done | the pivot of a new echelon row is its largest monomial (Python: the first dict key); the reduced basis does not depend on it |
| `polys/berlekampmassey.py` | done | generic over `Domain`; the bindings run it over every domain kind |
| `polys/bigatti.py` | done | moved out of `hilbert.rs` as in Python (5786e1d..eb367b4), with `hilbert_numerator_coefficients` |
| `polys/hilbert.py` | done | the numerator dense over `ZZ`, the series a `HilbertSeries { numerator, exponent }` (Python: `Poly` and an expression in `t`) |
| `polys/groebnerwalk.py` | done | both liftings; the bases of the initial forms by the Rust F4 (Python: SymPy's Buchberger, the same reduced bases); over fields only (SymPy also walks over `ZZ`) |
| `solvers/exponential_diophantine.py` | done | the `mpmath` steps on `dashu-float` at mpmath's binary precision for the same decimal digits; the bounds keep their margins, the partial quotients are checked at twice the digits |
| `solvers/simplex.py` | done | the first module of `solvers`; the status an enum (a string in the bindings); the rows of `_without_redundant` in the dict's insertion order |
| `ntheory/primetest.py`, `generate.py`, `factor_.py` | done | `primerange` is an iterator struct (a list in the bindings) |
| `polys/modulargroebner.py` | done | on the typed rings as the Python now is (7686069): `direct_groebner` with the algorithm as an argument (`settings` is not ported; the images use F4), `homogeneous_order`; `groebner` (the time-limited choice through `_timeout.attempt`) is not ported, it needs an interruptible F4 |
| `polys/buchberger.py` | done | ordered sets instead of Python sets (the reduced basis is unique) |
| `polys/polyclasses.py` | `DensePoly` | generic over the domain, the variable a core `Symbol`; binary operations return `Err(Value)` for different domains or variables; `from_expr`/`to_expr` wait for the core; no binding (it needs the core's `Symbol` in Python) |
| `polys/multisqf.py` | done | the `SqfDomain` classes as a trait with three implementations |
| `polys/multifactor.py` | done | the diophantine equations over `BigFiniteField` (the modulus is the next prime above a Mignotte bound) |
| `polys/gaussianfactor.py` | done | with `polytools._factor_all` as `factor_all`; `PolyRing` over `QQ_I` in the bindings; `to_expr` waits for the core |
| (new) `polys/domains/bigfinitefield.rs` | | `GF(p)` with `Int` elements for moduli beyond the machine words: Python's one `FiniteField` is split in two |
| `core/assumptions.py` | the rules and the structural facts | `Predicate` enum (alphabetical, so sorted as the names), implications, closure, consistency, minimal generators; `constant_facts`, `add_facts`, `mul_facts`, `pow_facts`, `numeric_facts`, `facts_of` (cached on the node), `ask_structural`, from the Python file at 2d17798. Two calls to evaluating functions are structural until they are ported: the consecutive-factor test (`g - f` an odd integer) and the irrationality of a rational power (`integer_root_parts`). `sympy_facts`/the function classes and `check_assumptions` & co. later |
| `core/sorting.py` | `compare` | the `compare` key; `sort_key`/`ordered` come with `Expr` |
| the bridge (`sympy_extras_rs.core`) | done | Python core expressions to Rust nodes by class name and arguments and back through the unevaluating classes, dummies kept in a registry both ways; `facts`, `is_number`, `numeric_value`, `compare`, `equal`, `free_symbols` compared on random trees (`tests/core`) |
| `core/add.py`, `core/mul.py`, `core/power.py` | the evaluation | `add`, `sub`, `mul`, `div`, `neg`, `keep_coeff`, `pow`, `number_pow`, `minus_one_pow`; the branches which build functions not ported (`exp`, `Abs`, `re`, `floor`, the relationals), `Float`, `Order` and the bounds are `PolyError::NotImplemented` (Python's `NotImplementedError`); 1181 random operations equal to Python's, structure and classes included (`tests/core`) |
| `core/printing.py`, `core/sorting.py` (sort keys) | printing | `sstr` (`str(e)`), `ordered_terms`, `precedence`, `could_extract_minus_sign`, `sort_key` (`default_sort_key`), `as_ordered_factors`; 11332 random evaluated expressions print as in Python (`tests/core`) |
| `functions/elementary/exponential.py` | `Exp`, `Log` | `exp`, `log`, `log(x, b)`, their facts, values and printing; `Head::Function`; `E**x` and `exp(x)*exp(y)` in `pow`/`mul`; `exp(oo*z)` of a number, the arctangent table and `expand_mul` of `log` are `NotImplemented`; 11939 random operations equal to Python's, facts included (`tests/core`) |
| `core/function.py` (expansion), `simplify/radsimp.py` (`fraction`) | expansion | `expand_with` and its `Hints` (`mul`, `multinomial`, `power_base`, `power_exp`, `log`, `force`; `func`, `trig`, `complex` not ported), `expand_mul`, `expand_multinomial`, `expand_default`, `expand_log_node`, `rebuild_evaluated` (`_evaluated`), `fraction`; 4712 random expansions over the hints equal to Python's; random expressions and hand picked fractions equal to Python's (`tests/core`) |
| `functions/elementary/complexes.py` (first part), `core/relational.py` (the ordering decision) | `Re`, `Im`, `Conjugate`, `Sign`, `Abs` | `re`, `im`, `conjugate`, `sign`, `abs_`, `as_real_imag`, `conjugate_of`, their facts and values; `compare`/`lt_true`; the `Abs`, `re` and `im` branches of `pow` and `exp`; what needs `cos`/`sin`/`atan2`, `arg`, `floor` or the full `expand` is `NotImplemented`; 14893 random operations equal to Python's, facts included (`tests/core`) |
| `functions/elementary/trigonometric.py` (direct functions) | `Sin`, `Cos`, `Tan`, `Cot`, `Sec`, `Csc` | the evaluation (exact values at rational multiples of `pi`, peeling multiples of `pi/2` off sums), facts, values, conjugates, real and imaginary parts of real arguments; `exp.as_real_imag`; `expand_default` (the default `expand` when the power and log hints have nothing to act on); the hyperbolic branches and the accumulation bounds are `NotImplemented`; 20424 exact values and random operations equal to Python's (`tests/core`) |
| `functions/elementary/hyperbolic.py` (direct functions) | `Sinh`, `Cosh`, `Tanh`, `Coth`, `Sech`, `Csch` | the evaluation (numbers, imaginary arguments to and from the trigonometric functions, multiples of `I*pi` peeled off sums), facts, values, conjugates, real and imaginary parts (also of the trigonometric functions of complex arguments); random operations equal to Python's (`tests/core`) |
| `polys/polyutils.py` | expressions and terms | `dict_from_expr`, `expr_from_dict`; `Domain::to_expr`/`from_expr` for `ZZ`, `QQ`, `GF(p)`, `ZZ_I`, `QQ_I` (the other domains: `NotImplemented`); random expressions equal to Python's over the five domains (`tests/polys`) |
| `core/exprtools.py` (main part) | terms and their content | `decompose_power`, `Factors`, `Term`, `gcd_terms`, `factor_terms`, `as_content_primitive` (numbers, sums, products, powers), `keep_coeff_with` (`clear`, `sign`); `factor_nc`, `ExprFactors` and `radical=True` are not ported; 66000 random checks equal to Python's, options included (`tests/core`) |
| `polys/rationaltools.py` | `together` | the sums over common denominators bottom up through `gcd_terms`, `deep` and `fraction`; random sums of quotients equal to Python's (`tests/polys`) |
| `core/function.py` (derivatives), `core/containers.py` (`Tuple`) | `AppliedUndef`, `Tuple`, `Derivative`, `Subs` | `UndefinedFunction` and `f(x)` (facts from the assumptions and the arguments), `diff` (sums, products, powers, the chain rule with the `fdiff` of every ported function, the own rules of `re`, `im`, `Abs`, `conjugate`, the undefined functions with `Derivative` and `Subs` at non-symbol points), their printing and order; `subs` and `sign`'s `DiracDelta` are `NotImplemented`; random derivatives equal to Python's (`tests/core`) |
| `core/basic.py` (`subs`), `core/power.py` (`power_subs`) | substitution | `subs`, `subs_pairs`, `subs_mapping` (SymPy's order of the keys), the replacement of sums and products among terms and factors, SymPy's rules for powers and `exp`, the bound variables of `Derivative` and `Subs`; `_substituted` now substitutes; 22377 random substitutions equal to Python's (`tests/core`) |
| `solvers/linsolve.py` | linear systems | `solve_linear_unique` (the reduced echelon form over `QQ(other symbols)`), `linear_root`, `solve_one`, `numeric_linear_solution`; `to_expr`/`from_expr` of the sparse polynomial rings and their fraction fields (`from_expr` by `as_numer_denom`: Python's `cancel` gives the same reduced element); the docstring examples and random systems equal to Python's (`tests/solvers`) |
| `polys/differential/ring.py` | differential rings | `DifferentialRing` (block rankings, jets, `terms`, `integral`, `leader`, `from_terms`, the coefficient field `QQ(p, x)`), `monomial_product`, `integer_product`, `integer_quotient`; `orderedmap::OrderedMap` for the Python dicts whose order reaches the results; `core::function::doit`; an `Equality` is given as `lhs - rhs`; the examples, the errors and random differential polynomials equal to Python's (`tests/polys`) |
| `polys/differential/polynomial.py` | differential polynomials | `DifferentialPolynomial` (arithmetic, `primitive`, leader, degree, rank, coefficients, initial, separant, cached derivatives, `prolonged`, Ritt's `pseudo_remainder`, `partial_remainder`, `remainder`); random polynomials and chains equal to Python's (`tests/polys`) |
| `polys/differential/janet.py` | Janet bases | `janet_basis` (autoreduction, completion for Janet's division, integrability conditions), `JanetBasis` (equations, leaders, multiplicative variables, `reduce`, `contains`, parametric derivatives, Hilbert function, series and polynomial, dimension, `series_solution` with the guarded `expand_default`), `janet_multiplicative_variables`; Janet's example, inconsistent and elimination systems and random systems equal to Python's (`tests/polys`) |
| `polys/functionfields.py` | function fields | `FunctionField` (generic in the constants: `QQ`, number fields, `QQ(params)`), `Element` arithmetic, inverse, derivative, trace, Trager's `integral_basis` (trace radical, idealizer, Hermite form), `local_basis_at_infinity`, `normal_basis`, pole orders, `function_field`, `integral_basis`; `to_expr`/`from_expr` of the dense polynomial rings and their fraction fields; the docstring curves, a parametric one and random curves equal to Python's (`tests/polys`) |
| `concrete/pisigma.py` | ΠΣ-fields | `PiSigmaField` (the tower and the field in one generic struct over `RationalConstants`/`ParametricConstants`), the shift and its inverse, `as_poly_in`, `level_of`, Karr's `solve` (constants, the rational level with Abramov's universal denominator and the dispersion by factors, Σ and Π recursion), `telescope`, `add_sigma`, `add_pi`; six towers (harmonic numbers, `2**k`, `k!`, `binomial(n, k)`) equal to Python's (`tests/concrete`) |
| `core/expr.py` | the decompositions | `as_coeff_Mul`, `as_base_exp`, `terms_of`, `factors_of`, `primitive`, `coeff`, `as_coeff_Add`, `as_numer_denom`, `extract_multiplicatively`, `as_coefficient`, `as_independent`, `is_comparable`; the `is_*` properties are the structural facts, as `Expr._ask` |
| `core/basic.py` | the structure | `Basic` = `Arc<Node>` with a `Head` per class (cached hash and compare key), `rebuild`, `atoms`, `free_symbols`, `has`, structural `xreplace`; `subs`/`doit`/`match`/`replace` come with the evaluating functions |
| `core/numbers.py` | exact numbers | `Integer`/`Rational` (one head, the class kept), infinities, `nan`, constants, `I`; `Float`, `AlgebraicNumber` and the number arithmetic next |
| `core/evalf.py` | `numeric_value` | on `dashu-float`/`dashu-cmplx` 0.6 (MIT/Apache) for `mpmath`; `EulerGamma` (Brent-McMillan), `Catalan` (Ramanujan), golden ratio and tribonacci constant computed here; `z**w` as `exp(w ln z)` because `dashu-cmplx` 0.6.2's `powf` hangs on exact results with a zero part (`(-1)**(1/2)`); `evalf`/`N` and `Float` next; a power beyond the machine-word exponents of dashu's floats (`2**(10**30)`) has no value (mpmath's exponents are unbounded) |
| `core/symbol.py` | the classes | `Symbol`, `Dummy`, `Wild` heads with typed views; `symbols()` next |
| everything else | not yet | the core continues: `expr`, `add`/`mul`/`power` (the evaluating functions), `printing`; then the polynomial `from_expr`/`to_expr` |
