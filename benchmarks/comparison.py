"""Side by side comparison of sympy-extras and SymPy.

For every entry of :data:`CASES` the same mathematical question is put to
SymPy alone and to sympy-extras, both under a time limit, and the two
answers are printed (as a table, or as the Markdown of
``docs/comparison.md`` with ``--markdown``). Nothing here is asserted by
hand: the report is generated from what the two libraries actually
return, so it can be regenerated after a SymPy release with

    python benchmarks/comparison.py --markdown > /tmp/comparison.txt

The ``verdict`` of a case says what SymPy does with the question:

``error``
    it raises (``NotImplementedError``, ``ValueError``, ...);
``undecided``
    it returns the question unevaluated (``Sum(...)``, ``ConditionSet``,
    ``None``, an unchanged expression);
``wrong``
    it returns an answer which is not correct (checked in the ``check``
    field of the case);
``partial``
    it answers, but less completely than sympy-extras.
"""
from __future__ import annotations

import sys
import time
import traceback
from typing import Optional

from sympy_extras._timeout import time_limit, TimeLimitExceeded

TIMEOUT = 60.0


class Case:
    """One question, asked of both libraries."""

    def __init__(self, section: str, title: str, extras: str, sympy: str, verdict: str,
                 note: str = '', check: str = '') -> None:
        self.section = section
        self.title = title
        self.extras = extras
        self.sympy = sympy
        self.verdict = verdict
        self.note = note
        self.check = check
        self.extras_result = ''
        self.sympy_result = ''
        self.check_result = ''
        self.extras_time = 0.0
        self.sympy_time = 0.0


def _run(code: str, namespace: dict[str, object]) -> tuple[str, float]:
    started = time.monotonic()
    try:
        with time_limit(TIMEOUT):
            value = eval(code, namespace)          # noqa: S307 - the code is written here
        text = str(value)
    except TimeLimitExceeded:
        text = 'no answer within %gs' % TIMEOUT
    except Exception as error:                      # noqa: BLE001 - every failure is data
        text = '%s: %s' % (type(error).__name__, str(error).split('\n')[0][:120])
    return text, time.monotonic() - started


PREAMBLE = """
from sympy import *
from sympy.abc import a, b, c, k, m, n, p, q, s, t, u, v, w, x, y, z
import sympy_extras
from sympy_extras.assumptions import (ask as xask, refine as xrefine, simplify as xsimplify, resolve,
    solve as xsolve, element,
    ForAll, Exists, satisfiable, tautology, find_instance, limit as xlimit, limit_seq as xlimit_seq,
    series as xseries)
from sympy_extras.concrete import (karr_sum, summation as xsummation, zeilberger_sum, wz_prove,
    sum_convergence, product_convergence, dirichlet_series, polygamma_series, euler_sum,
    qgosper_sum, qzeilberger, rational_sum)
from sympy_extras.solvers import (dsolve_kovacic, dsolve_linear, special_solutions, dsolve_lie,
    dsolve_first_order, dsolve_second_order, dsolve_dae, dae_matrices, complete_integral,
    isolate_real_roots, thue, abel_by_invariants, abel_invariants, linear_diophantine_system,
    minimal_nonnegative_solutions, cooper, dsolve_linear_system, symmetries, pde_symmetries,
    rational_system_solutions, integrating_factor_xy, is_linearizable)
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.groebnerwalk import groebner_walk
from sympy_extras.polys.euclidtools import psc
from sympy_extras.polys.cad import quantifier_elimination, solution_set
from sympy_extras.polys.virtual_substitution import eliminate_linear
from sympy_extras.concrete.qhyper import qpochhammer, qbinomial
from sympy_extras.integrals import definite_integral, IntegralByRanges, mellin_transform as xmellin
from sympy.solvers.ode import dsolve
from sympy.series.limitseq import limit_seq
from sympy.polys import ring, QQ, ZZ, grevlex, lex
from sympy.polys.groebnertools import groebner as ring_groebner
from sympy.logic.inference import satisfiable as sympy_satisfiable
from sympy_extras.solvers import pdsolve_lie

f = Function('y')(x)
g = Function('u')(x, y)
y1, y2 = Function('y1')(x), Function('y2')(x)


def implicit_residual(equation, solution, points):
    'The residual of an implicit solution F(x, y) = C1, by implicit differentiation.'
    from sympy.solvers.solvers import solve as _s
    relation = solution.lhs - solution.rhs
    [rhs] = _s(equation, f.diff(x))
    total = relation.diff(x).subs(f.diff(x), rhs)
    return [N(total.subs(f, y0).subs(x, x0).subs(Symbol('C1'), 0), 10) for x0, y0 in points]


R3, X, Y, Z = ring("x,y,z", QQ, grevlex)
R1, X3 = ring("x", ZZ)
RA, A, B = ring("a,b", ZZ)
RAB, X3, A, B = ring("x,a,b", ZZ)
"""


CASES: list[Case] = [
    # ------------------------------------------------------------- assumptions
    Case('Assumptions and real algebra',
         'Quantifier elimination over the reals',
         "resolve(ForAll(x, x**2 + b*x + c > 0))",
         "'no counterpart'",
         'error',
         'The cylindrical algebraic decomposition is the only complete decision procedure for '
         'the elementary theory of the reals, and SymPy has none.'),
    Case('Assumptions and real algebra',
         'The condition for a quadratic to have a real root',
         "resolve(Exists(x, Eq(a*x**2 + b*x + c, 0)))",
         "'no counterpart'",
         'error'),
    Case('Assumptions and real algebra',
         'A polynomial inequality under polynomial assumptions',
         "xask(x**2 + y**2 >= 2*x*y, (x > 0) & (y > 0))",
         "ask(Q.nonnegative(x**2 + y**2 - 2*x*y), Q.positive(x) & Q.positive(y))",
         'undecided'),
    Case('Assumptions and real algebra',
         'An inequality between elementary functions',
         "xask(sin(x) < x, x > 0)",
         "ask(Q.negative(sin(x) - x), Q.positive(x))",
         'undecided',
         'Decided from polynomial (MetiTarski style) bounds handed to the CAD.'),
    Case('Assumptions and real algebra',
         'The exponential bound',
         "xask(exp(x) >= 1 + x, element(x, S.Reals))",
         "ask(Q.nonnegative(exp(x) - 1 - x), Q.real(x))",
         'undecided'),
    Case('Assumptions and real algebra',
         'Reality of a square root under an assumption on the parameter',
         "xask(element(sqrt(a - 2), S.Reals), a > 0)",
         "ask(Q.real(sqrt(a - 2)), Q.positive(a))",
         'wrong',
         "SymPy's ``Q.real`` handler for powers does not look at the sign of the base, so it "
         'claims that the root is real for every positive ``a``. It is not, at ``a = 1``:',
         "sqrt(S(1) - 2)"),
    Case('Assumptions and real algebra',
         'Refining an expression with two absolute values',
         "xrefine(Abs(x - 1) + sqrt(x**2), x > 2)",
         "refine(Abs(x - 1) + sqrt(x**2), Q.positive(x - 2))",
         'undecided'),
    Case('Assumptions and real algebra',
         'A square root of a perfect square',
         "xrefine(sqrt(x**2 - 2*x + 1), x > 1)",
         "refine(sqrt(x**2 - 2*x + 1), Q.positive(x - 1))",
         'undecided'),
    Case('Assumptions and real algebra',
         'Parity of a polynomial in an integer variable',
         "xrefine((-1)**(n**2 + n), element(n, S.Integers))",
         "refine((-1)**(n**2 + n), Q.integer(n))",
         'undecided'),
    Case('Assumptions and real algebra',
         'A residue of an integer polynomial',
         "xrefine(Mod(n**3 - n, 6), element(n, S.Integers))",
         "refine(Mod(n**3 - n, 6), Q.integer(n))",
         'undecided'),
    Case('Assumptions and real algebra',
         'Simplifying a relation under assumptions',
         "xsimplify(Eq(x**2, 1), x > 0)",
         "refine(Eq(x**2, 1), Q.positive(x))",
         'undecided'),
    Case('Assumptions and real algebra',
         'Satisfiability of a nonlinear real system',
         "satisfiable((x**2 + y**2 < 1) & (x*y > Rational(1, 2)))",
         "sympy_satisfiable((x**2 + y**2 < 1) & (x*y > Rational(1, 2)))",
         'wrong',
         "SymPy's SAT solver is propositional: it treats each inequality as an opaque literal "
         'and returns a "model" of a system which has no real solution '
         '(``x**2 + y**2 >= 2*Abs(x*y) > 1``).'),
    Case('Assumptions and real algebra',
         'A witness for a nonlinear real system',
         "find_instance((x**2 + y**2 < 1) & (x*y > Rational(1, 5)), [x, y])",
         "'no counterpart'",
         'error'),

    # ------------------------------------------------------------- solving
    Case('Solving equations',
         'A logarithmic equation over the reals',
         "xsolve(Eq(log(x) + log(x - 1), log(2)), x, domain=S.Reals)",
         "solveset(Eq(log(x) + log(x - 1), log(2)), x, S.Reals)",
         'wrong',
         'Neither logarithm is real at ``x = -1``.',
         "[(log(v) + log(v - 1)).doit() for v in (-1, 2)]"),
    Case('Solving equations',
         'A radical equation',
         "xsolve(2*x**2 + 3*sqrt(x + 6) - 1, x, domain=S.Reals)",
         "solveset(2*x**2 + 3*sqrt(x + 6) - 1, x, S.Reals)",
         'wrong',
         'Squaring without checking gives four candidates whose membership in the reals is left '
         'unevaluated. Two of them are not roots of the equation at all (their residuals are '
         '17.17 and 12.17) and the other two are not real, so the real solution set is empty.',
         "[N(2*r**2 + 3*sqrt(r + 6) - 1, 8) for r in "
         "[c for c in solveset(2*x**2 + 3*sqrt(x + 6) - 1, x, S.Reals).args if c.is_FiniteSet][0]]"),
    Case('Solving equations',
         'An equation with no closed form solution',
         "xsolve(Eq(x, cos(x)), x, domain=S.Reals)",
         "solveset(Eq(x, cos(x)), x, S.Reals)",
         'undecided',
         'The Dottie number, as an exact root object which evaluates to any precision.',
         "N(list(xsolve(Eq(x, cos(x)), x, domain=S.Reals))[0], 25)"),
    Case('Solving equations',
         'The real roots of a transcendental function',
         "isolate_real_roots(exp(x) - x - 2, x)",
         "solveset(exp(x) - x - 2, x, S.Reals)",
         'undecided',
         'Two roots, isolated in intervals proved to contain exactly one root each.'),
    Case('Solving equations',
         'An inequality with transcendental endpoints',
         "xsolve(cos(x) - x**2/4 <= 0, x, domain=S.Reals)",
         "solveset(cos(x) - x**2/4 <= 0, x, S.Reals)",
         'undecided'),
    Case('Solving equations',
         'Both real branches of a Lambert equation',
         "xsolve(exp(x) - x - 2, x, domain=S.Reals)",
         "solve(exp(x) - x - 2, x)",
         'partial',
         'SymPy returns the principal branch only, so the equation looks as if it had one real '
         'root instead of two.'),
    Case('Solving equations',
         'A parametric equation with a sign assumption',
         "xsolve(x**2 - a, x, (x > 0) & (a > 0))",
         "solveset(x**2 - a, x, S.Reals)",
         'partial'),
    Case('Solving equations',
         'A conjunction of polynomial inequalities',
         "xsolve((x**3 - 2*x > 0) & (x < 3), x, domain=S.Reals)",
         "solveset((x**3 - 2*x > 0) & (x < 3), x, S.Reals)",
         'error'),
    Case('Solving equations',
         'A system of linear Diophantine equations',
         "linear_diophantine_system([Eq(2*x + 3*y + 5*z, 7), Eq(x + y + z, 2)], [x, y, z])",
         "diophantine(2*x + 3*y + 5*z - 7)",
         'partial',
         '``diophantine`` takes one equation at a time; there is no solver for systems.'),
    Case('Solving equations',
         'Minimal nonnegative solutions (Contejean-Devie)',
         "minimal_nonnegative_solutions([[3, -5]], [1])",
         "'no counterpart'",
         'error'),
    Case('Solving equations',
         'Presburger arithmetic',
         "resolve(ForAll(x, Exists(y, Eq(3*y, x) | Eq(3*y, x + 1) | Eq(3*y, x + 2))), domain=S.Integers)",
         "'no counterpart'",
         'error'),
    Case('Solving equations',
         "A Thue equation (Thomas's cubic)",
         "thue(x**3 + x**2*y - 2*x*y**2 - y**3, 1, x, y)",
         "diophantine(x**3 + x**2*y - 2*x*y**2 - y**3 - 1)",
         'error',
         "All nine solutions, and proved to be all of them by Baker's method."),
    Case('Solving equations',
         "A Thue equation of Nagell",
         "thue(x**3 - 2*y**3, 1, x, y)",
         "diophantine(x**3 - 2*y**3 - 1)",
         'error'),
    Case('Solving equations',
         'Reduction over the complex numbers',
         "resolve(Exists(x, Eq(a*x**2 + b*x + c, 0) & Ne(x, 0)), domain=S.Complexes)",
         "'no counterpart'",
         'error'),

    # ------------------------------------------------------------- ODEs
    Case('Differential equations',
         "Kovacic's algorithm on Weber's equation",
         "dsolve_kovacic(f.diff(x, 2) - (x**2 + 3)*f, f)",
         "dsolve(f.diff(x, 2) - (x**2 + 3)*f, f)",
         'partial',
         'SymPy returns six terms of a power series; the Liouvillian solutions are elementary.',
         "[simplify(u.diff(x, 2) - (x**2 + 3)*u) for u in dsolve_kovacic(f.diff(x, 2) - (x**2 + 3)*f, f)]"),
    Case('Differential equations',
         'A linear equation with an exponential solution',
         "dsolve_kovacic(x*f.diff(x, 2) + (1 - x)*f.diff(x) - f, f)",
         "dsolve(x*f.diff(x, 2) + (1 - x)*f.diff(x) - f, f)",
         'partial'),
    Case('Differential equations',
         'Polynomial and hyperexponential solutions of a linear ODE',
         "dsolve_linear(x*f.diff(x, 2) - (x + 2)*f.diff(x) + 2*f, f)",
         "dsolve(x*f.diff(x, 2) - (x + 2)*f.diff(x) + 2*f, f)",
         'partial',
         'Abramov-Bronstein-Petkovsek: a polynomial solution and a hyperexponential one, exactly.'),
    Case('Differential equations',
         'A Whittaker equation',
         "special_solutions(f.diff(x, 2) + (-Rational(1, 4) + 1/x + (Rational(1, 4) - n**2)/x**2)*f, f)",
         "dsolve(f.diff(x, 2) + (-Rational(1, 4) + 1/x + (Rational(1, 4) - n**2)/x**2)*f, f)",
         'partial',
         'Recognised through the normal form invariant; SymPy falls back on a power series.'),
    Case('Differential equations',
         'A second order equation solved through commuting symmetries',
         "dsolve_second_order(f.diff(x, 2) + 3*f*f.diff(x) + f**3, f)",
         "dsolve(f.diff(x, 2) + 3*f*f.diff(x) + f**3, f)",
         'error',
         "Bocharov's example: linearisable, but not by a fibre preserving transformation.",
         "checkodesol(f.diff(x, 2) + 3*f*f.diff(x) + f**3, dsolve_second_order(f.diff(x, 2) + 3*f*f.diff(x) + f**3, f)[0])"),
    Case('Differential equations',
         "Lie's linearisation test",
         "is_linearizable(-3*y*p - y**3, x, y, p)",
         "'no counterpart'",
         'error'),
    Case('Differential equations',
         'An Abel equation of the first kind',
         "abel_by_invariants(f.diff(x) + f**3 + 2*(x - 1)*f**2, f) is not None",
         "dsolve(f.diff(x) + f**3 + 2*(x - 1)*f**2, f)",
         'error',
         'Recognised by its invariants as equivalent to the Airy class. The implicit solution is '
         'a long expression in Bessel functions; the residuals of its implicit differentiation '
         'along the equation are:',
         "implicit_residual(f.diff(x) + f**3 + 2*(x - 1)*f**2, "
         "abel_by_invariants(f.diff(x) + f**3 + 2*(x - 1)*f**2, f), ((0.6, 0.4), (1.1, 0.9)))"),
    Case('Differential equations',
         'A Chini equation',
         "dsolve_first_order(f.diff(x) - f**3 - x**Rational(-3, 2), f) is not None",
         "dsolve(f.diff(x) - f**3 - x**Rational(-3, 2), f)",
         'error',
         "Chini's invariant is constant, so the equation separates. The residuals of the "
         'implicit solution are:',
         "implicit_residual(f.diff(x) - f**3 - x**Rational(-3, 2), "
         "dsolve_first_order(f.diff(x) - f**3 - x**Rational(-3, 2), f), ((0.6, 0.4), (1.1, 0.9)))"),
    Case('Differential equations',
         'A Riccati equation without a rational particular solution',
         "dsolve_first_order(f.diff(x) - f**2 - x, f).rhs.atoms(besselj)",
         "dsolve(f.diff(x) - f**2 - x, f)",
         'error',
         'The general solution in Bessel functions. SymPy needs a rational particular solution '
         'and crashes with a ``TypeError`` when there is none.'),
    Case('Differential equations',
         'Point symmetries of a second order equation',
         "[X.generator() for X in symmetries(f.diff(x, 2) + 2*f.diff(x)/x + f**5, f)]",
         "classify_ode(f.diff(x, 2) + 2*f.diff(x)/x + f**5, f)",
         'error',
         "SymPy's ``lie_group`` hint is for first order equations only."),
    Case('Differential equations',
         'A second order equation solved through its symmetry',
         "dsolve_lie(f.diff(x, 2) - f.diff(x)**2/f - f.diff(x)/x, f)",
         "dsolve(f.diff(x, 2) - f.diff(x)**2/f - f.diff(x)/x, f)",
         'error'),
    Case('Differential equations',
         'A differential-algebraic system',
         "dsolve_dae(Matrix([[1, 0], [0, 0]]), Matrix([[0, -1], [1, 0]]), Matrix([0, sin(x)]), x).solution.T",
         "dsolve([Eq(y1.diff(x), y2), Eq(y1, sin(x))], [y1, y2])",
         'error',
         'A pencil of index 2: the solution is unique, with no free constant. SymPy raises a '
         '``KeyError``.'),
    Case('Differential equations',
         'A linear system with rational coefficients',
         "[Y.T for Y in dsolve_linear_system(Matrix([[0, 1], [2/x**2, 0]]), x)]",
         "dsolve([Eq(y1.diff(x), y2), Eq(y2.diff(x), 2*y1/x**2)], [y1, y2])",
         'error'),
    Case('Differential equations',
         'A complete integral of a nonlinear first order PDE',
         "complete_integral(g.diff(x)*g.diff(y) - 1, g)",
         "pdsolve(g.diff(x)*g.diff(y) - 1, g)",
         'error',
         "Charpit's method; ``pdsolve`` handles first order linear equations only."),
    Case('Differential equations',
         'Symmetries of the heat equation',
         "[X.generator() for X in pde_symmetries(g.diff(y) - g.diff(x, 2), g, degree=2)]",
         "'no counterpart'",
         'error'),
    Case('Differential equations',
         'Group invariant solutions of the heat equation',
         "pdsolve_lie(g.diff(y) - g.diff(x, 2), g)",
         "pdsolve(g.diff(y) - g.diff(x, 2), g)",
         'error',
         'Every solution is verified with ``checkpdesol`` before it is returned.'),

    # ------------------------------------------------------------- sums
    Case('Sums, products and series',
         'An indefinite sum of harmonic numbers',
         "karr_sum(harmonic(k), (k, 1, n))",
         "summation(harmonic(k), (k, 1, n))",
         'undecided',
         "Karr's algorithm in the Pi-Sigma field built from the summand."),
    Case('Sums, products and series',
         'A sum of squares of harmonic numbers',
         "karr_sum(harmonic(k)**2, (k, 1, n))",
         "summation(harmonic(k)**2, (k, 1, n))",
         'undecided'),
    Case('Sums, products and series',
         'A nested sum',
         "karr_sum(harmonic(k)/(k + 1), (k, 1, n))",
         "summation(harmonic(k)/(k + 1), (k, 1, n))",
         'undecided'),
    Case('Sums, products and series',
         "Dixon's identity by creative telescoping",
         "zeilberger_sum((-1)**k*binomial(2*n, k)**3, (k, 0, 2*n))",
         "summation((-1)**k*binomial(2*n, k)**3, (k, 0, 2*n))",
         'partial',
         "Zeilberger's algorithm. SymPy returns an unevaluated hypergeometric function, not a "
         'closed form.'),
    Case('Sums, products and series',
         'A Wilf-Zeilberger certificate',
         "wz_prove(binomial(n, k)**2, binomial(2*n, n), n, k)",
         "'no counterpart'",
         'error'),
    Case('Sums, products and series',
         'Convergence of a power series with a parameter',
         "sum_convergence(x**n/n, n)",
         "Sum(x**n/n, (n, 1, oo)).is_convergent()",
         'error',
         'The exact condition, boundary point included.'),
    Case('Sums, products and series',
         'Convergence of the p-series',
         "sum_convergence(1/n**p, n)",
         "Sum(1/n**p, (n, 1, oo)).is_convergent()",
         'error'),
    Case('Sums, products and series',
         'Convergence of an alternating series with a parameter',
         "sum_convergence((-1)**n*n**p, n)",
         "Sum((-1)**n*n**p, (n, 1, oo)).is_convergent()",
         'error'),
    Case('Sums, products and series',
         'Convergence of an infinite product',
         "product_convergence(1 + x/n**2, n)",
         "Product(1 + x/n**2, (n, 1, oo)).is_convergent()",
         'error'),
    Case('Sums, products and series',
         'An Euler sum',
         "polygamma_series(harmonic(n)/n**2, n)",
         "Sum(harmonic(n)/n**2, (n, 1, oo)).doit()",
         'undecided'),
    Case('Sums, products and series',
         'A series of polygamma functions',
         "polygamma_series(polygamma(1, n)/n**2, n)",
         "Sum(polygamma(1, n)/n**2, (n, 1, oo)).doit()",
         'undecided'),
    Case('Sums, products and series',
         'The Dirichlet series of Euler totient',
         "dirichlet_series(totient(n)/n**s, n)",
         "Sum(totient(n)/n**s, (n, 1, oo)).doit()",
         'undecided'),
    Case('Sums, products and series',
         'The Dirichlet series of the Moebius function',
         "dirichlet_series(mobius(n)/n**s, n)",
         "Sum(mobius(n)/n**s, (n, 1, oo)).doit()",
         'undecided'),
    Case('Sums, products and series',
         'A q-hypergeometric sum and a q-recurrence',
         "(factor(qgosper_sum(q**k, (k, 0, n), q)), "
         "qzeilberger(qbinomial(n, k, q)*q**(k*(k - 1)/2)*x**k, n, k, q).coefficients)",
         "'no counterpart'",
         'error',
         'q-Gosper and q-Zeilberger (the q-binomial theorem); SymPy has no q-analogues.'),

    # ------------------------------------------------------------- limits
    Case('Limits and series expansions',
         'A limit depending on the sign of a parameter',
         "xlimit(exp(a*x), x, oo)",
         "limit(exp(a*x), x, oo)",
         'error',
         'The cases are returned as a ``Piecewise``, the counterpart of Mathematica '
         '``GenerateConditions``.'),
    Case('Limits and series expansions',
         'A limit under an assumption',
         "xlimit(x**a*log(x), x, 0, assumptions=a > 0)",
         "limit(x**a*log(x), x, 0)",
         'error'),
    Case('Limits and series expansions',
         'A sequence limit with a parameter',
         "xlimit_seq(a**n, n, assumptions=a > 0)",
         "limit_seq(a**n, n)",
         'undecided',
         '``exp(oo*sign(log(a)))`` is not an answer.'),
    Case('Limits and series expansions',
         'An oscillating sequence',
         "xlimit_seq(a**n, n, assumptions=(a > -1) & (a < 0))",
         "limit_seq(a**n, n)",
         'undecided'),
    Case('Limits and series expansions',
         'A series expansion under an assumption',
         "xseries(sqrt(a**2 + x), x, 0, 2, assumptions=a < 0)",
         "sqrt(a**2 + x).series(x, 0, 2)",
         'partial',
         'SymPy cannot resolve ``sqrt(a**2)``.'),

    # ------------------------------------------------------------- polynomials
    Case('Polynomials and ideals',
         'An elimination ideal',
         "Ideal([x - t**2, y - t**3], t, x, y).eliminate([t])",
         "'no counterpart'",
         'error',
         "SymPy's ``agca`` ideals have no elimination, intersection, quotient or saturation."),
    Case('Polynomials and ideals',
         'A saturation',
         "Ideal([x*z - y**2, x**2 - y*z], x, y, z).saturate(Ideal([y], x, y, z))",
         "'no counterpart'",
         'error'),
    Case('Polynomials and ideals',
         'The radical of an ideal',
         "Ideal([x**2 - 2*x*y + y**2, x**3], x, y).radical()",
         "'no counterpart'",
         'error'),
    Case('Polynomials and ideals',
         'The Hilbert series',
         "Ideal([x**2, x*y], x, y).hilbert_series()",
         "'no counterpart'",
         'error'),
    Case('Polynomials and ideals',
         'The dimension and degree of an ideal',
         "(Ideal([x*z - y**2, x**2 - y*z], x, y, z).dimension(), "
         "Ideal([x*z - y**2, x**2 - y*z], x, y, z).degree())",
         "'no counterpart'",
         'error'),
    Case('Polynomials and ideals',
         'A Groebner basis conversion in positive dimension',
         "groebner_walk(ring_groebner([X*Z - Y**2, X**2 - Y*Z], R3), R3, lex)",
         "groebner([x*z - y**2, x**2 - y*z], x, y, z, order='grevlex').fglm('lex')",
         'error',
         "SymPy's FGLM is restricted to zero-dimensional ideals; the walk works in any dimension."),
    Case('Polynomials and ideals',
         'Principal subresultant coefficients',
         "psc(X3**3 + A*X3 + B, 3*X3**2 + A)",
         "'no counterpart'",
         'error',
         'SymPy has ``subresultants`` (the polynomials) but not their principal coefficients, '
         'which are what the CAD projection operators need. The last one is the discriminant.'),
    Case('Polynomials and ideals',
         'Linear quantifier elimination by virtual substitution',
         "eliminate_linear((x > y) & (x < 1), x)",
         "'no counterpart'",
         'error'),
]


CASES += [
    # ------------------------------------------------------------ integration
    Case('Definite integration',
         'A singularity inside the range',
         "definite_integral(1/x, (x, -1, 2))",
         "integrate(1/x, (x, -1, 2))",
         'undecided',
         'The integral diverges; SymPy returns `nan`. `definite_integral` cuts the range at the '
         'singularities it finds and claims nothing when a piece diverges, so the unevaluated '
         'integral comes back.'),
    Case('Definite integration',
         'Powers of trigonometric functions: Beta integrals',
         "definite_integral(sqrt(sin(x)), (x, 0, pi/2))",
         "integrate(sqrt(sin(x)), (x, 0, pi/2))",
         'undecided',
         'The substitution `x = asin(sqrt(u))` turns powers of `sin` and `cos` over a quarter '
         'period into a Beta integral, one kernel of the Mellin table.'),
    Case('Definite integration',
         'A periodic integrand with kinks',
         "definite_integral(sqrt(1 - cos(x)), (x, 0, 2*pi))",
         "integrate(sqrt(1 - cos(x)), (x, 0, 2*pi))",
         'undecided',
         '`1 - cos(x)` is `2*sin(x/2)**2` and the square root is `sqrt(2)*|sin(x/2)|`.'),
    Case('Definite integration',
         'Logarithms and powers on the unit interval',
         "definite_integral(x**Rational(1, 3)/sqrt(-log(x)), (x, 0, 1))",
         "integrate(x**Rational(1, 3)/sqrt(-log(x)), (x, 0, 1))",
         'undecided',
         '`(-log(x))**k` on `(0, 1)` has the Mellin transform `gamma(k + 1)/s**(k + 1)`.'),
    Case('Definite integration',
         'Integrands in exp(x) over the real line',
         "definite_integral(x*exp(x)*exp(k*x)/(exp(x) + 3), (x, -oo, oo), (k > -1) & (k < 0))",
         "integrate(x*exp(x)*exp(k*x)/(exp(x) + 3), (x, -oo, oo))",
         'undecided',
         'The substitution `u = exp(x)` gives `log(u)*u**k/(u + 3)` over `(0, oo)`: a power of '
         '`x` times a kernel, and the logarithm is a derivative with respect to the exponent.'),
    Case('Definite integration',
         'A Laplace transform with a product of two kernels',
         "definite_integral(exp(-s*x)*sin(a*x)/x, (x, 0, oo), (s > 0) & (a > 0))",
         "integrate(exp(-s*x)*sin(a*x)/x, (x, 0, oo))",
         'partial',
         'Parseval\'s formula for the Mellin transform turns the product into a Meijer '
         'G-function, which Slater\'s theorem expands; the assumptions decide the conditions. '
         'SymPy returns a `Piecewise` with conditions on `arg(a)` and `arg(s)` and the value '
         'written with `sqrt(a**2/s**2)`.'),
    Case('Definite integration',
         'The Mellin transform of atan',
         "xmellin(atan(x), x, s)",
         "mellin_transform(atan(x), x, s)",
         'undecided',
         'The table of transforms with their strips of convergence; SymPy has no entry for `atan`.'),
    Case('Definite integration',
         'An integral over a region described by inequalities',
         "IntegralByRanges(x*y, (x > 0) & (y > 0) & (x + y < 1)).doit()",
         "'no counterpart'",
         'error',
         'The region is decomposed into stacks of intervals by the cylindrical algebraic '
         'decomposition and the iterated integrals are computed innermost first, the way '
         'Mathematica\'s `Integrate[f, {x, y} ∈ region]` works.'),
    Case('Definite integration',
         'The area of a disc of parametric radius',
         "IntegralByRanges(1, x**2 + y**2 < c**2, [x, y]).doit()",
         "'no counterpart'",
         'error',
         'The parameter is the first variable of the decomposition and gives the case distinction.'),
]


def run(cases: list[Case]) -> None:
    namespace: dict[str, object] = {}
    exec(PREAMBLE, namespace)                       # noqa: S102 - the code is written here
    for case in cases:
        case.extras_result, case.extras_time = _run(case.extras, namespace)
        case.sympy_result, case.sympy_time = _run(case.sympy, namespace)
        if case.check:
            case.check_result, _ = _run(case.check, namespace)


def _shorten(text: str, width: int) -> str:
    text = ' '.join(text.split())
    return text if len(text) <= width else text[:width - 3] + '...'


def table(cases: list[Case]) -> None:
    section: Optional[str] = None
    for case in cases:
        if case.section != section:
            section = case.section
            print('\n== %s' % section)
        print('  %-52s %-8s %6.1fs' % (_shorten(case.title, 52), case.verdict, case.extras_time))
        print('      extras: %s' % _shorten(case.extras_result, 150))
        print('      sympy : %s' % _shorten(case.sympy_result, 150))
        if case.check_result:
            print('      check : %s' % _shorten(case.check_result, 150))


def markdown(cases: list[Case]) -> None:
    section: Optional[str] = None
    for case in cases:
        if case.section != section:
            section = case.section
            print('\n## %s\n' % section)
        print('### %s\n' % case.title)
        print('```python')
        print('>>> %s' % case.extras)
        print('%s' % _shorten(case.extras_result, 240))
        print('```\n')
        print('SymPy: `%s` -> `%s`\n' % (case.sympy, _shorten(case.sympy_result, 240)))
        if case.note:
            print('%s\n' % case.note)


def main() -> None:
    run(CASES)
    if '--markdown' in sys.argv:
        markdown(CASES)
    else:
        table(CASES)


if __name__ == '__main__':
    try:
        main()
    except Exception:                               # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
