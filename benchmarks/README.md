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
| `solve_random.py` | random polynomial equations with sign assumptions | `solve` against `Poly.real_roots` and high-precision evaluation |

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
