# sympy-extras

Extensions to [SymPy](https://www.sympy.org): algorithms built on top of
SymPy which are not (yet) part of SymPy itself.

SymPy is a large, conservative project: getting a new algorithm merged means
meeting its review standards, keeping every corner case of a stable public
API working and waiting for a release cycle. **sympy-extras** is the place
for algorithms that are useful today but do not fit that process yet: they
live here, mirror the layout of SymPy's own modules, are tested against
released SymPy versions, and may move into SymPy proper later.

## Policy

- **An extension, not a fork.** `sympy-extras` only depends on SymPy and
  adds new functionality on top of it. It does not patch or replace anything
  in SymPy.
- **A lax policy on AI-generated algorithms.** Code written with the help of
  AI models (or entirely by them) is welcome here, and a large part of the
  code in this repository was generated that way. What is required is the
  same as for any other code: a clear description of the algorithm with
  references, docstrings with examples, and tests which check the results
  against independent sources (hand computations, known results, other
  computer algebra systems). Provenance of AI-generated code is stated in
  commit messages, not hidden.
- **No support against breaking changes as of now.** The project is at
  version 0.x. Any release may rename, move or remove public functions and
  change their results; there is no deprecation policy yet. Pin the exact
  version if you depend on it. Changes are listed in
  [CHANGELOG.md](CHANGELOG.md).

## Installation

```
pip install sympy-extras

```

`sympy-extras` requires Python 3.9 or later and SymPy 1.14 or later.

To work on the code, clone the repository and install it in editable mode
with the test dependencies:

```
pip install -e ".[test]"
python -m pytest

```

The test command runs both the unit tests and the doctests in the
docstrings.

## Contents

### Cylindrical algebraic decomposition (`sympy_extras.polys.cad`)

A cylindrical algebraic decomposition (CAD) of $\mathbb{R}^n$ adapted to a
set of polynomials in $x_1, \ldots, x_n$ is a partition of $\mathbb{R}^n$
into finitely many connected cells on each of which every polynomial has a
constant sign. Each cell comes with an exact sample point, so any property
that only depends on the signs of the polynomials can be decided by looking
at finitely many points. This is the basis of Collins' decision procedure
and quantifier elimination for the first order theory of the real numbers.

```python
>>> from sympy import Eq
>>> from sympy.abc import a, b, c, x, y
>>> from sympy_extras.polys.cad import cylindrical_algebraic_decomposition
>>> cad = cylindrical_algebraic_decomposition([x**2 + y**2 - 1], [x, y])
>>> cad
CAD(13 cells, x, y)
>>> [cell.point for cell in cad if cell.signs == (0,)]
[(-1, 0), (0, -1), (0, 1), (1, 0)]

```

Quantifier elimination, decision of closed formulas, solution sets and
sample points of systems of polynomial equations and inequalities are built
on top of the decomposition:

```python
>>> from sympy_extras.polys.cad import quantifier_elimination, decide, solution_set, sample_points
>>> quantifier_elimination(x**2 + b*x + c > 0, [('forall', x)])
b**2 - 4*c < 0
>>> quantifier_elimination(Eq(x**2 + a*x + b, 0), [('exists', x)])
a**2 - 4*b >= 0
>>> solution_set(Eq(x**2 + y**2, 1) & (y > x), x, [('exists', y)])
Interval.Ropen(-1, CRootOf(2*x**2 - 1, 1))
>>> decide(Eq(y, x**2), [('forall', x), ('exists', y)])
True
>>> sample_points((x**2 + y**2 < 1) & (x > y), [x, y])
[{x: 0, y: -1/2}, {x: CRootOf(2*x**2 - 1, 1), y: 0}, {x: 3/4, y: 0}]

```

The implementation follows the classical two phases, projection (McCallum's
operator by default, Hong's as a fallback when the input is not
well-oriented) and lifting with exact real algebraic sample points kept in a
single algebraic number field. See [docs/cad.md](docs/cad.md) for a longer
description and the API reference.

The number of cells grows quickly with the number of variables and the
degrees: the implementation is meant for problems with a few variables and
moderate degrees.

### Assumptions as mathematical statements (`sympy_extras.assumptions`)

An alternative front end to SymPy's assumptions, modelled on Mathematica's
user interface with Python names. Assumptions are written as ordinary
statements instead of predicates: `x > 0` for `Q.positive(x)`,
`element(n, S.Integers)` (that is `Contains(n, S.Integers)`) for
`Q.integer(n)`, intervals and other sets, combined with `&`, `|`, `~`, and
quantified with `ForAll` and `Exists`. SymPy's `ask`, `refine`, `simplify`
and SAT solver are the backends, together with the cylindrical algebraic
decomposition above for everything polynomial over the reals.

```python
>>> from sympy import S, Abs, sqrt, Eq
>>> from sympy.abc import b, c, x, y, n
>>> from sympy_extras.assumptions import ask, refine, element, resolve, satisfiable, ForAll, Exists
>>> ask(x**2 - 2*x + 1 >= 0, x > 0)
True
>>> ask(x > 1, x > 2)
True
>>> refine(Abs(x - 1) + sqrt(x**2), x > 2)
2*x - 1
>>> ask(element(n**2 + n, S.Integers), element(n, S.Integers))
True
>>> resolve(ForAll(x, x**2 + b*x + c > 0))
b**2 - 4*c < 0
>>> resolve(Exists(y, Eq(x**2 + y**2, 1) & (y > x)))
(x >= -1) & (x < CRootOf(2*x**2 - 1, 1))
>>> satisfiable((x**2 + y**2 < 1) & (x + y > 1))
{x: 1/2, y: 2/3}

```

`refine`, `simplify`, `ask`, `assuming`/`global_assumptions`, `resolve`,
`satisfiable`, `tautology` and `find_instance` correspond to Mathematica's
`Refine`, `Simplify`, `Assuming`/`$Assumptions`, `Resolve`, `SatisfiableQ`,
`TautologyQ` and `FindInstance`. See
[docs/assumptions.md](docs/assumptions.md).

### Principal subresultant coefficients (`sympy_extras.polys.euclidtools`)

`dup_psc`, `dmp_psc` and `psc` compute the principal subresultant
coefficients of two polynomials, the leading coefficients of the
subresultant sequence, which are needed by Hong's projection operator.

```python
>>> from sympy import ring, ZZ
>>> from sympy_extras.polys.euclidtools import psc
>>> R, x, y = ring("x,y", ZZ)
>>> psc(x**2*y + x, x + y)
[y**3 - y, 1]

```

## Layout

The package mirrors the layout of SymPy: code extending `sympy.polys` lives
in `sympy_extras/polys`, and so on. Tests live next to the code in `tests`
subdirectories and use the same conventions as SymPy's tests.

```
sympy_extras/
    assumptions/
        facts.py             assumptions as statements, translation to predicates and polynomials
        quantifiers.py       ForAll, Exists, prenex normal form
        context.py           assuming, global_assumptions
        ask.py               ask
        refine.py            refine, simplify
        resolve.py           resolve (quantifier elimination)
        sat.py               satisfiable, tautology, find_instance
    polys/
        euclidtools.py       principal subresultant coefficients
        cad/
            projection.py    projection operators (McCallum, Hong)
            samplepoints.py  exact real algebraic sample points
            lifting.py       lifting phase, cylindrical_algebraic_decomposition
            qe.py            quantifier elimination and decision

```

## Releasing

Releases are published to PyPI by the `Release` GitHub Actions workflow
when a tag `vX.Y.Z` is pushed. To release:

1. Update `__version__` in `sympy_extras/__init__.py` and add a section to
   `CHANGELOG.md`.
2. Commit, then tag and push:
   ```
   git tag v0.0.1
   git push origin master v0.0.1
   ```
3. The workflow checks that the tag matches the version, runs the tests,
   builds the sdist and wheel, publishes them to PyPI with trusted
   publishing and creates a GitHub release.

The workflow needs a one-time setup: a `pypi` environment in the repository
settings, and the workflow registered as a trusted publisher for the
`sympy-extras` project on PyPI (see the comment at the top of
`.github/workflows/release.yml`). A release can also be built and uploaded
by hand with `python -m build` and `python -m twine upload dist/*`.

## License

BSD 3-Clause, see [LICENSE](LICENSE).
