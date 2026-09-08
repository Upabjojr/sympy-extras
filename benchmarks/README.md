# Benchmarks

Drivers which run the algorithms of sympy-extras on external collections
with known answers. They are not part of the test suite: they take minutes
to hours and some download their data on first use into `benchmarks/.cache/`
(ignored by git; nothing of the downloaded files is stored in the
repository).

| Driver | Data | Checks |
|---|---|---|
| `pde_symmetries.py` | 12 classical PDEs with the dimension of their point symmetry algebra from Olver, Bluman-Kumei and Ibragimov's handbook | dimension of the algebra found with the stated ansatz; every symmetry verified by substitution |
| `kamke_odes.py` | the Kamke collection of first and second order ODEs (about 700 equations) as transcribed in the test suite of Maxima's `contrib_ode` (downloaded from SourceForge) | `dsolve` and, when it fails, `dsolve_lie`; every solution verified with `checkodesol` under a time limit |
| `solve_random.py` | random polynomial equations with sign assumptions; with `--transcendental N`, random equations in `exp`, `log`, `sin`, `cos`, `sqrt` on random intervals | `solve` against `Poly.real_roots`, or against sign changes on a fine grid refined with `nsolve`, and high-precision evaluation |
| `verify_random.py` | a random sample of the Kamke collection | `solve_ode` (`dsolve`, then the symmetry method); every solution verified with `checkodesol` and numerically (implicit solutions by implicit differentiation and `nsolve`); a solution failing the numerical check is reported as WRONG |
| `logic_random.py` | random Boolean combinations of polynomial relations in one or two real variables with random assumptions | `simplify`, `refine`, `ask`, `satisfiable` against evaluation at random points satisfying the assumptions; for one variable the CAD decides the equivalence of the simplified formula |
| `qf_nra.py` | SMT-LIB `QF_NRA`, Meti-Tarski family (7713 problems with `:status`, cloned sparsely from the `dreal/benchmarks` mirror on GitHub) | `satisfiable` against the status (models re-evaluated); `simplify` and `refine` over the reals against the status and random points |

## Results (SymPy 1.14, one core, 15 s per step)

`pde_symmetries.py`: 12 of 12 equations give the expected dimension and
all symmetries verify (about 10 s in total).

`kamke_odes.py --collections kamke1 --limit 60 --timeout 15` (the first
60 first order equations, mostly linear, Riccati and Abel equations):

| | verified | unverified | failed | timeout |
|---|---|---|---|---|
| `dsolve` | 23 | 9 | 15 | 13 |
| `dsolve` then `dsolve_lie` | 24 | 9 | 15 | 12 |

`dsolve_lie` alone on the 37 equations `dsolve` did not verify: 1
verified, 4 unverified (a solution which `checkodesol` could not confirm
within the limit), 30 failed, 2 timeouts.

`linear_odes.py --limit 60 --timeout 10` (the first 60 homogeneous linear
equations of the collections, 3 of first order and 57 of second order,
almost all with symbolic parameters), after adding Bessel/Whittaker/
hypergeometric recognition, exponential parts from the Newton polygon and
the numerical check of hypergeometric solutions:

| verified | verified numerically | partial | failed |
|---|---|---|---|
| 37 | 18 | 5 | 0 |

"Verified numerically" are solutions with `hyper` which `checkodesol`
cannot simplify (it even returns `False` for them: issue #25), checked at
three points with 20 digits; "partial" are the five equations where only
one solution of the basis was found (the second one needs a `log` term
or a `2F1` second solution at an integer exponent difference). Before the
special functions were added the same run gave 28 verified, 2 partial and
30 failed.

`kamke_odes.py --collections kamke1 --limit 60 --timeout 15` with the
Abel/Chini/Lagrange solvers tried after `dsolve` ("extras"): `dsolve` 23
verified; extras add 1 (a Chini equation), `dsolve_lie` 2; combined 26
verified, 9 unverified, 15 failed, 10 timeouts. The first order part of
Kamke is dominated by Riccati and Abel equations without a constant
invariant, which no closed-form method solves.

`kamke_odes.py --collections kamke2 --limit 40 --timeout 15` (the first 40
second order equations, almost all linear with special function
solutions):

| | verified | unverified | failed | timeout |
|---|---|---|---|---|
| `dsolve` | 11 | 16 | 12 | 1 |
| `dsolve` then `dsolve_lie` | 11 | 16 | 12 | 1 |

The polynomial ansatz finds no useful symmetry of a linear equation whose
symmetries involve its own solutions, which is what these are; the
symmetry method pays off on nonlinear equations (see the unit tests and
`docs/solvers.md`). "Unverified" means a solution was produced but
`checkodesol` did not confirm it within the time limit; it is not counted
as solved.

`solve_random.py --cases 150`: 150 of 150 agree with the oracle.

`verify_random.py --sample 24 --seed 7 --timeout 15`: 16 verified, 5
unverified, 3 failed, **0 wrong** solutions.

`solve_random.py --cases 100 --transcendental 40 --seed 11` and
`--cases 60 --transcendental 30 --seed 4`: 230 of 230 agree with the
oracles. (The first runs found mismatches: two bugs of the oracle, at an
open endpoint and at a root of even multiplicity, and an extraneous root
returned by SymPy's `solveset` for the radical equation
`2*x**2 + 3*sqrt(x + 6) - 1 = 0`, which `solve` now drops by checking every
candidate against the equation; complex candidates are dropped when real
solutions are wanted.)

`logic_random.py --cases 80 --seed 3` (one variable) and
`--cases 40 --seed 5 --variables 2`: 0 problems, 7 cases skipped (no
point satisfying the random assumption was found).
`logic_random.py --transcendental 40 --seed 9` (random expressions in
`exp`, `log`, `sin`, `cos`, `sqrt`, `atan` of one variable on random
intervals, `ask(f > 0)` and `refine(Abs(f))` against dense sampling):
0 wrong, 15 undecided (a sign change inside the interval, where `None` is
the right answer, or the zeros out of reach of `solveset`).
`logic_random.py --bivariate 30 --seed 21 --timeout 60` (random
expressions of two variables on random boxes, decided by interval branch
and bound, monotonicity at the corners, the range of an inner argument
and the polynomial bounds handed to the CAD): 0 wrong, 15 undecided.

`qf_nra.py --sample 20 --seed 2 --timeout 20 --max-variables 3`: no
wrong answer; `satisfiable` decided 8 of 20 (12 undecided within the
limit), `simplify` and `refine` finished 7 of 20 (13 timeouts). The
Meti-Tarski problems have 3 variables and coefficients with 7-8 digits,
which is where this CAD is slow (#10).
