"""Point symmetry algebras of classical partial differential equations,
compared with the dimensions published in the literature.

Usage::

    python benchmarks/pde_symmetries.py

For every equation the finite-dimensional part of the point symmetry
algebra (the symmetries which only add solutions of a linear equation are
excluded) is computed with a polynomial ansatz of the degree given in the
table, its dimension compared with the reference, and every symmetry
verified by substitution in the invariance condition.

References: P. J. Olver, *Applications of Lie Groups to Differential
Equations* (1993), chapter 2 and the exercises; G. W. Bluman, S. Kumei,
*Symmetries and Differential Equations* (1989); N. H. Ibragimov (ed.),
*CRC Handbook of Lie Group Analysis of Differential Equations*, vol. 1
(1994).
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sympy import Function, symbols, exp, sin, log, Rational

from sympy_extras.solvers import pde_symmetries, check_symmetry

x, t, m, a = symbols('x t m a')
u = Function('u')(x, t)

#: (name, equation, degree of the ansatz, extra basis functions, expected
#: dimension with this ansatz, note with the literature value)
TABLE = [
    ("heat u_t = u_xx", u.diff(t) - u.diff(x, 2), 3, (), 6,
     "Olver ex. 2.41: 6 generators plus the superposition ideal"),
    ("Burgers u_t + u u_x = u_xx", u.diff(t) + u*u.diff(x) - u.diff(x, 2), 2, (), 5,
     "Olver ex. 2.43 (Galilean algebra)"),
    ("KdV u_t + u u_x + u_xxx = 0", u.diff(t) + u*u.diff(x) + u.diff(x, 3), 2, (), 4,
     "Olver ex. 2.44"),
    ("wave u_tt = u_xx", u.diff(t, 2) - u.diff(x, 2), 2, (), 7,
     "polynomial part of the conformal algebra: translations, boost, dilation, 2 special conformal, u d/du"),
    ("nonlinear diffusion u_t = (u^m u_x)_x", u.diff(t) - (u**m*u.diff(x)).diff(x), 2, (), 4,
     "Ovsiannikov's classification: 4 for generic m"),
    ("nonlinear diffusion m = -4/3", u.diff(t) - (u**Rational(-4, 3)*u.diff(x)).diff(x), 2, (), 5,
     "the exceptional exponent: an extra projective symmetry"),
    ("Fisher u_t = u_xx + u(1 - u)", u.diff(t) - u.diff(x, 2) - u*(1 - u), 2, (), 2,
     "translations only"),
    ("Boussinesq u_tt + u u_xx + u_x^2 + u_xxxx = 0",
     u.diff(t, 2) + u*u.diff(x, 2) + u.diff(x)**2 + u.diff(x, 4), 2, (), 3,
     "translations and a scaling (Clarkson-Kruskal)"),
    ("Liouville u_xt = exp(u)", u.diff(x, t) - exp(u), 2, (), 6,
     "infinite algebra f(x) d/dx + g(t) d/dt - (f' + g') d/du: 3 + 3 polynomials of degree <= 2"),
    ("sine-Gordon u_xt = sin(u)", u.diff(x, t) - sin(u), 2, (), 3,
     "translations and the Lorentz-type scaling x d/dx - t d/dt"),
    ("potential Burgers u_t = u_xx + u_x^2", u.diff(t) - u.diff(x, 2) - u.diff(x)**2, 3, (), 6,
     "linearisable by u = log v: the heat algebra"),
    ("Black-Scholes-type u_t + x^2 u_xx/2 = 0", u.diff(t) + x**2*u.diff(x, 2)/2, 2, (log(x),), 5,
     "5 of the 6 generators of Gazizov-Ibragimov; the sixth needs t*log(x)**2 terms"),
]


def main() -> int:
    failures = 0
    print("%-46s %7s %5s %5s %s" % ("equation", "degree", "found", "ref", "time"))
    for name, eq, degree, basis, reference, note in TABLE:
        started = time.time()
        syms = pde_symmetries(eq, u, degree=degree, basis=basis)
        elapsed = time.time() - started
        verified = all(check_symmetry(eq, u, s) for s in syms)
        status = "ok" if len(syms) == reference and verified else "MISMATCH"
        if status != "ok":
            failures += 1
        print("%-46s %7d %5d %5d %5.1fs %s  %s" % (name, degree, len(syms), reference, elapsed, status, note))
    print("mismatches:", failures)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
