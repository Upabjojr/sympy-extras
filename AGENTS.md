# Guidance for AI agents and contributors

This file describes how to work on `sympy-extras`. It is meant for AI coding
agents as well as for people.

## What this project is

`sympy-extras` is an **extension to [SymPy](https://www.sympy.org)**: a
collection of algorithms built on top of SymPy which are not (yet) part of
SymPy itself. It depends on released SymPy versions and adds functionality;
it never monkeypatches or replaces anything inside SymPy.

The project has a **lax policy on AI-generated algorithms**: code written
with AI assistance, or entirely by AI models, is welcome. Much of the code
in this repository was produced that way. The bar is the same as for any
other code, and it is about correctness and verifiability rather than
authorship:

- the algorithm is described in the module docstring with references to
  the literature;
- every public function has a docstring with doctested examples;
- there are tests which check the results against independent sources
  (hand computations, known results, other computer algebra systems such as
  QEPCAD or Redlog), not only against the code's own output. Results
  published in the literature or in the documentation of other systems
  may be used as test data (they are facts), but their code must not be
  copied: the licenses do not allow it. Such tests live in the
  `test_*_known.py` files;
- **every bug which is found gets a unit test which fails before the fix
  and passes after it**, in the `tests/` directory next to the code (see
  *Regression tests* below);
- the provenance of the code (which model, which pull request it was ported
  from) is stated in the commit message.

The project is at version 0.x and **does not guarantee support against
breaking changes** for now. Public functions may be renamed, moved or
removed in any release. Every user-visible change goes into `CHANGELOG.md`.

## Standing instructions from the maintainer

These were given while the project was being built and apply to every
change:

- **Never copy code** from SymPy, Mathematica, Maxima, Maple or any other
  system: their licenses do not allow it. Published results, test data
  and algorithm descriptions are facts and may be used; implementations
  are written from the literature and stated in the module docstring
  with references.
- **Algorithms that need no change stay in SymPy.** SymPy is a dependency:
  call it (`solveset`, `dsolve`, `groebner`, `roots`, `gosper_sum`,
  `limit`, ...) and only add what is missing or must be rewritten
  (assumptions as statements, the CAD, decision procedures). The tables
  in `docs/reduce.md` say for each algorithm of Mathematica's
  implementation notes whether it lives in SymPy, here, or nowhere yet.
- **Assumptions are mathematical statements** (`x > 0`, `element(x,
  S.Integers)`), not `Q` predicates or `Symbol` flags; every function
  that takes assumptions accepts them in this form.
- **Verify against independent sources at scale**: datasets and
  benchmark drivers live in `benchmarks/` (Kamke's ODEs from Maxima's
  test suite, SMT-LIB QF_NRA, random equations against numerical
  oracles); random samples are pulled from them and wrong results are
  fixed, not worked around. Results go into `benchmarks/README.md`.
- **Numerical checks are optional**: `sympy_extras.settings` has a global
  switch (`numerical_checks`), the precision and the time limit, with
  the `configure` context manager; exact results must not depend on
  them, they only prune candidates and verify.
- **Report SymPy bugs and limitations in this repository's issue
  tracker** (issue #25 collects them with snippets and the workaround
  used here), so that they can be reported upstream; work around them
  in this package rather than waiting.
- **Time limits everywhere**: SymPy routines that may not terminate run
  under `sympy_extras._timeout.attempt` with the limit of the settings.
- **Merge into `master`**: this is a private experimental repository, the
  development branch is fast-forwarded into `master` after every
  verified change (tests, doctests, mypy, pyflakes); the changelog and
  the docs are updated in the same commit.
- **Track the remaining work in issues**: what is not implemented or
  only partially (issues #23 and #24 for the Mathematica lists) is
  written down there with the reasons, and updated when progress is
  made.

## Layout

The package mirrors SymPy's layout: code extending `sympy.polys` goes in
`sympy_extras/polys`, code extending `sympy.solvers` would go in
`sympy_extras/solvers`, and so on. This makes it straightforward to move a
module into SymPy later.

```
sympy_extras/
    __init__.py              __version__
    assumptions/             assumptions written as mathematical statements
        facts.py             element(), normalize(), Facts: translation to predicates and polynomial relations
        quantifiers.py       ForAll, Exists, prenex()
        context.py           assuming(), global_assumptions
        ask.py               ask(): sympy.ask first, then the CAD
        refine.py            refine(), simplify(): sympy.refine plus CAD-decided handlers
        resolve.py           resolve(): quantifier elimination through the CAD
        sat.py               satisfiable(), tautology(), find_instance(): SAT + theory check
        tests/
    concrete/                summation algorithms extending sympy.concrete
        pisigma.py           PiSigmaField: the tower, sigma, the parameterized first order solver
        karr.py              summand analysis (build_pisigma_field), karr_sum, karr_term, summation
        tests/
    polys/
        euclidtools.py       principal subresultant coefficients (dup_psc, dmp_psc, psc)
        ideals.py            Ideal: operations and invariants on top of sympy.groebner
        groebnerwalk.py      groebner_walk, extended_groebner (Buchberger with representations)
        orderings.py         WeightOrder, BlockOrder, elimination_order (hashable monomial orders)
        tests/
        cad/                 cylindrical algebraic decomposition
            projection.py    projection operators (McCallum, Hong), projection_sets
            samplepoints.py  exact real algebraic sample points (SamplePoint)
            lifting.py       lifting phase, cylindrical_algebraic_decomposition, CAD, CADCell
            qe.py            quantifier_elimination, decide, sample_points, solution_set
            tests/
docs/                        longer documentation per module
.github/workflows/           tests.yml (CI), release.yml (PyPI publishing on tags)
conftest.py                  pytest configuration (SymPy-style doctest display)
pyproject.toml               package metadata; version is read from sympy_extras/__init__.py
```

Tests live next to the code in `tests/` subdirectories, as in SymPy, and
are shipped with the package.

## Conventions

- Follow SymPy's coding conventions: the low level polynomial routines use
  the dense `dup_*`/`dmp_*` representation, the high level ones take `Expr`
  or `Poly`; use `Poly`, `CRootOf`, the domains in `sympy.polys.domains`
  and SymPy's own containers rather than reinventing them.
- Only use public or well-established SymPy APIs and check that the code
  runs on the minimum supported SymPy version declared in `pyproject.toml`
  (SymPy master and released versions differ; for instance
  `sympy.polys.densebasic.dmp_zero` takes one argument in releases).
- Docstrings use the SymPy style: a summary, `Parameters`, `Returns`,
  `Examples` (with `>>>` doctests) and `References` sections with `====`
  underlines.
- Doctests are run with `pytest --doctest-modules` and follow SymPy's
  conventions: results are displayed with `str`, so `QQ(2, 5)` prints as
  `2/5`. `conftest.py` installs the display hook.
- Tests use plain `assert` statements and `sympy.testing.pytest.raises`
  for exceptions, like SymPy's tests.
- Keep the code pure Python with SymPy (and mpmath, through SymPy) as the
  only runtime dependency.
- Do not reimplement what SymPy already does: call SymPy's algorithms
  (`ask`, `refine`, `simplify`, `satisfiable`, the polynomial routines) and
  add what is missing on top. Algorithms that need no change stay in SymPy.

## Types: strict type stability (required)

Type stability is a hard requirement of this project, not a style
preference: the code must be translatable to a statically typed language
(C++ or Rust) one day, so every value must have one explicit, statically
known type. mypy in strict mode enforces it (`python -m mypy`, configured
in `pyproject.toml`) and the CI job fails on any error; the package ships a
`py.typed` marker.

Rules, all of them mandatory for every change:

1. **Every function, method, lambda-free callable and test has explicit
   parameter and return annotations.** No untyped or partially typed
   definitions; tests return `None` and their helpers are annotated too.
2. **No implicit `Any`.** `Any` is allowed only for elements of SymPy
   coefficient domains (`DomainElement`) and inside the dense polynomial
   aliases `Dup`/`Dmp`, both defined in `sympy_extras/_typing.py`. Do not
   introduce new uses of `Any`, `object` as a catch-all, or untyped
   containers (`list`, `dict`, `tuple` without parameters).
3. **Narrow at the SymPy boundary.** SymPy 1.14 ships no `py.typed` and
   few annotations, so its results are `Basic` or `Any`. The moment a
   SymPy value enters our code, convert it with the helpers of
   `sympy_extras/_typing.py` (`as_expr`, `as_boolean`, `as_symbol`,
   `as_set`, `free_symbols`, `sorted_symbols`), which check the class at
   run time and raise `TypeError`. Never use `typing.cast` to silence a
   SymPy type; never rely on duck typing across a union.
4. **Unions are narrowed with `isinstance` before use**, `Optional` values
   are checked before use, and a function returns one type of value (or a
   documented `Optional`), never a value whose type depends on an
   argument, except through `@overload` on a literal argument (as
   `satisfiable` does with `all_models`).
5. **Name the structures.** Recurring shapes get an alias in
   `sympy_extras/_typing.py` (`Monomial`, `Sign`, `Truth`, `Weights`,
   `OrderSpec`, `QuantifierPrefix`, ...) or in the module that owns them
   (`Solution`, `Rhs` in `pisigma.py`, `CellTruth` in `qe.py`, `Model`,
   `Witness` in `sat.py`). Use them instead of repeating tuple types.
6. **Attributes are declared with their type**, in `__init__` or as class
   level annotations, before being assigned in other methods
   (`PiSigmaField` declares `field`, `gens`, `zero`, ... in `__init__`).
   Attributes that SymPy sets dynamically (so that mypy cannot see them)
   are not read at all: narrow the object with `isinstance` to the class
   that declares the attribute (`_unit` narrows to `AlgebraicField`), or
   rebuild the value from a typed API (`_root_poly` builds the polynomial
   of a `ComplexRootOf` from its expression; `PiSigmaField` gets its zero
   and one from `field(0)` and `field(1)`). `getattr` is forbidden.
7. **`# type: ignore`, `typing.cast` and `getattr` are forbidden.** The
   code base has none, and a change that adds one is not accepted:
   restructure the code so that the types are honest instead. Two
   consequences of this rule shape the API:
   - Constructors do not evaluate. A SymPy-style `__new__` that returns a
     different object in trivial cases cannot be typed, so `ForAll(x, True)`
     is a `ForAll` instance and the trivial cases are reduced by
     `simplify()`, `prenex` and the functions consuming quantified
     formulas. Follow the same pattern for new classes.
   - Tests which check that a wrong argument type raises `TypeError` call
     the function through `sympy_extras._testing.untyped`, whose type
     `Callable[..., object]` makes the mismatch explicit instead of
     suppressing it. A compiled formula in `qe.py` is a small hierarchy of
     node classes with typed fields, not a tagged union.
8. **The mypy configuration is not to be weakened.** `strict = true` stays;
   the only relaxations are `disallow_untyped_calls` and
   `warn_return_any` (both forced off by SymPy's untyped API) and
   `strict_equality` for the tests (they compare SymPy numbers with Python
   literals). Do not add `ignore_errors`, per-module `disallow_*` = false,
   or new `ignore_missing_imports` entries beyond `mpmath`.
9. **Run mypy before every commit.** `python -m mypy` must report
   `Success: no issues found`; a change with type errors is not done, and
   the CI job (`typecheck` in `.github/workflows/tests.yml`) rejects it.

When SymPy gains annotations, tighten the configuration (enable
`warn_return_any`, then `disallow_untyped_calls`) rather than relying on
the relaxations. Issue #15 tracks what cannot be checked yet and the
planned typed core layer independent of SymPy objects.

## Regression tests for every bug

A bug which is fixed without a test comes back. Whenever a wrong result, a
crash, a hang or a state leak is found in this package — by a user, by a
benchmark driver, by a doctest of the documentation, or while working on
something else — the fix comes with a test:

1. **Write the test first**, as the smallest input which shows the wrong
   behaviour, and check that it fails on the unfixed code. A test which
   passes before the fix is testing something else.
2. **Put it next to the code it covers**, in the `tests/` directory of the
   subpackage, in the test function which covers that feature or in a new
   one named after the symptom (`test_precision_is_restored`,
   `test_extraneous_roots_are_dropped`).
3. **Say what the bug was**, in one comment line above the assertion: what
   was returned before, and why it was wrong. The comment is what makes
   the test readable in a year.
4. **Check the property, not the printed form**, whenever the two differ:
   a solution is checked by substitution, a sum against its partial sums,
   a sign against a sample point. A test which pins a printed expression
   fails on every harmless change of form and gets weakened or deleted.
5. **A bug found in SymPy** (not in this package) gets an entry in the
   issue tracker and, if the package works around it, a test of the
   workaround with the SymPy behaviour quoted in the comment, so that the
   workaround can be removed when SymPy fixes it.

The randomised drivers in `benchmarks/` (`fuzz.py`, `solve_random.py`,
`logic_random.py`, `verify_random.py`, `qf_nra.py`) exist to find these
cases: they cross-check the results against independent oracles
(numerical evaluation, brute force search, substitution). When one of them
reports a failure, the fix and its unit test follow; the driver itself is
not a substitute for the test.

## Running the tests

```
pip install -e ".[test]"
python -m pytest
```

This runs the unit tests and the doctests of every module under
`sympy_extras/`. The examples in `README.md` and `docs/` are doctests too:

```
python -m pytest --doctest-glob='*.md' README.md docs
python -m mypy
python -m pyflakes sympy_extras conftest.py
```

Run everything before committing; a change is not done until the tests,
mypy and pyflakes pass.

## Adding a new algorithm

1. Put it in the subpackage matching the SymPy module it extends, with
   tests in the neighbouring `tests/` directory.
2. Export the public names from the subpackage `__init__.py` and list them
   in `__all__`.
3. Add a page in `docs/` if the module needs more explanation than the
   docstrings, and a short section with an example to `README.md`.
4. Add an entry to `CHANGELOG.md` under the upcoming version.
5. State the provenance in the commit message (AI model used, original
   pull request if the code was ported from a SymPy pull request).

## Releasing

Releases are cut from `master` by pushing a tag `vX.Y.Z` that matches
`__version__` in `sympy_extras/__init__.py`. The `Release` workflow checks
the version, runs the tests, builds the sdist and wheel, publishes them to
PyPI with trusted publishing and creates a GitHub release. Bump the version
and the changelog in a commit before tagging; never publish a version that
was already published. See the `Releasing` section of `README.md`.

## Things not to do

- Do not call a SymPy routine which may not terminate (`integrate`,
  `dsolve`, `solve`, `simplify` on unbounded input, `checkodesol`) without
  a time limit: wrap it with `sympy_extras._timeout.attempt`, as the
  solvers do, so that every public function returns.
- Benchmarks against external collections live in `benchmarks/`, download
  their data on first use into `benchmarks/.cache/` (never committed) and
  are not part of the test suite; the unit tests must stay fast.
- Do not change SymPy's behaviour from this package (no monkeypatching).
- Do not add dependencies beyond SymPy without a discussion.
- Do not delete or weaken a test to make the suite pass.
- Do not fix a bug without adding the test which fails on the unfixed code
  (see *Regression tests for every bug*).
- Do not silently drop the provenance of ported code.
- Do not add an unannotated function, an untyped container, a new `Any`,
  a `cast`, a `getattr` or a `# type: ignore`, and do not weaken the mypy
  configuration.
