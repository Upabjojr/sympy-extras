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
- the provenance of the code (which model, which pull request it was ported
  from) is stated in the commit message.

The project is at version 0.x and **does not guarantee support against
breaking changes** for now. Public functions may be renamed, moved or
removed in any release. Every user-visible change goes into `CHANGELOG.md`.

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

## Running the tests

```
pip install -e ".[test]"
python -m pytest
```

This runs the unit tests and the doctests of every module under
`sympy_extras/`. The examples in `README.md` and `docs/` are doctests too:

```
python -m pytest --doctest-glob='*.md' README.md docs
```

Run everything before committing; a change is not done until the tests
pass.

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

- Do not change SymPy's behaviour from this package (no monkeypatching).
- Do not add dependencies beyond SymPy without a discussion.
- Do not delete or weaken a test to make the suite pass.
- Do not silently drop the provenance of ported code.
