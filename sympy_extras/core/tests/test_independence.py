"""The core does not import SymPy: importing it, the elementary and special
functions, and running the usual operations leave ``sympy`` out of
``sys.modules`` (SymPy is lent to the core only by ``sympy_extras.interop``,
through :mod:`sympy_extras.core.bridge`)."""
from __future__ import annotations

import subprocess
import sys

_PROGRAM = r'''
import sys
from sympy_extras.core import symbols, Rational, sqrt, expand, diff, Eq, pi, factor_terms, I, Integer
from sympy_extras.core.limits import limit
from sympy_extras.core.bridge import ForeignOperationUnavailable
from sympy_extras.functions import sin, exp, log, gamma
x, y = symbols('x y')
results = [expand((x + y)**3/2), diff(sin(x)*exp(x), x), ((x + y)**2).subs(x, 2), factor_terms(2*x + 2*y),
           (sin(x)/x).series(x, 0, 6), limit(sin(x)/x, x, Integer(0)), gamma(Rational(1, 2)), log(exp(2)),
           pi.evalf(30), (x*I)**2, sqrt(8), Eq(x, x)]
assert str(results[4]) == '1 - x**2/6 + x**4/120 + O(x**6)', results[4]
from sympy_extras.core import sympify, SympifyError
assert str(sympify('sin(x)**2 + log(8, 2) - x/2')) == '-x/2 + sin(x)**2 + 3', sympify('sin(x)**2 + log(8, 2) - x/2')
try:
    sympify('x if y else z')   # outside the native grammar, and no bridge
except SympifyError:
    pass
else:
    raise AssertionError('should not parse without the bridge')
try:
    from sympy_extras.core import Symbol
    assert str(Symbol('x').integrate(Symbol('x'))) == 'x**2/2'   # native (term by term)
    (1/(Symbol('x') + 1)).integrate(Symbol('x'))
except ForeignOperationUnavailable:
    pass
else:
    raise AssertionError('integrate should need the bridge')
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_core_does_not_load_sympy() -> None:
    out = subprocess.run([sys.executable, '-c', _PROGRAM], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_the_bridge_is_registered_by_interop() -> None:
    import sympy_extras.interop as interop
    from sympy_extras.core.bridge import bridge
    from sympy_extras.core import sympify
    assert isinstance(bridge(), interop.SymPyBridge)
    assert str(sympify('x + 1')) == 'x + 1'


_NATIVE_SOLVERS = r'''
import sys
from sympy_extras.core import symbols
from sympy_extras.functions import binomial
from sympy_extras.solvers.linsolve import solve_linear_unique
from sympy_extras.solvers.recurr import rsolve_hypergeometric
from sympy_extras.concrete.creative import definite_sum
n, k, a, b = symbols('n k a b')
assert str(solve_linear_unique([a + b - 1, a - b - n], [a, b])) == '[n/2 + 1/2, 1/2 - n/2]'
assert str(definite_sum(binomial(n, k), (k, 0, n))) == '2**n'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_solvers_do_not_load_sympy() -> None:
    """The package ``sympy_extras.solvers`` exports nothing, so importing
    its native modules (and :mod:`sympy_extras.concrete.creative`, which
    uses them) does not load the ones written on SymPy."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_SOLVERS], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_ZEILBERGER = r'''
import sys
from sympy_extras.core import symbols
from sympy_extras.functions import binomial
from sympy_extras.concrete.zeilberger import zeilberger, zeilberger_sum, wz_prove
n, k = symbols('n k', integer=True)
assert str(zeilberger(binomial(n, k)**2, n, k)) == \
    'Telescoper([-2*(2*n + 1), n + 1], k**2*(2*k - 3*n - 3)/(k - n - 1)**2)'
assert str(zeilberger_sum(binomial(n, k)**2, (k, 0, n))) == 'binomial(2*n, n)'
assert str(zeilberger_sum(1/k, (k, 1, n))) == 'harmonic(n)'
assert str(zeilberger_sum(binomial(2*n, k), (k, 0, n))) == '4**n/2 + factorial(2*n)/(2*factorial(n)**2)'
assert wz_prove(binomial(n, k), 2**n, n, k) is True
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_zeilberger_does_not_load_sympy() -> None:
    """:mod:`sympy_extras.concrete.zeilberger` on a summand without
    parameters (the cases of the parameters are the only step through
    SymPy) does not load SymPy."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_ZEILBERGER], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_QHYPER = r'''
import sys
from sympy_extras.core import symbols
from sympy_extras.concrete.qhyper import qgosper_sum, qgosper_term, qzeilberger, qbinomial, qpochhammer
q, k, n, x, a = symbols('q k n x a')
assert str(qgosper_sum(q**k, (k, 0, n), q, special_values=False)) == 'q**(n + 1)/(q - 1) - 1/(q - 1)'
assert str(qgosper_term(q**k*qpochhammer(a, q, k), k, q, special_values=False)) == '-QPochhammer(a, q, k)/a'
assert str(qzeilberger(qbinomial(n, k, q)*q**(k*(k - 1)/2)*x**k, n, k, q).coefficients) == '[-q**n*x - 1, 1]'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_qhyper_does_not_load_sympy() -> None:
    """:mod:`sympy_extras.concrete.qhyper` without the cases of the
    parameters (``q`` is one: they always go through SymPy) does not load
    SymPy."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_QHYPER], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_ASSUMPTIONS = r'''
import sys
from sympy_extras.core import symbols, Q, ask
from sympy_extras.functions import exp
from sympy_extras.sets import Integers, interval
from sympy_extras.assumptions.facts import Facts, element, normalize, to_predicates, to_polynomial
from sympy_extras.assumptions.quantifiers import ForAll, Exists, forall, prenex
from sympy_extras.assumptions.bounds import polynomial_abstraction
from sympy_extras.assumptions.context import assuming, global_assumptions
x, y, z, a = symbols('x y z a')
f = Facts([x > 2, element(y, Integers), z**3 > x*y, x/a > 1])
assert str(f.polynomial) == '(x > 2) & (z**3 > x*y) & (a*(-a + x) > 0)', f.polynomial
assert f.integer == {y} and f.real == {x, y, z, a}
assert str(normalize(element(x, interval(0, 1)))) == '(x >= 0) & (x <= 1)'
assert str(to_predicates(x - y >= 2)) == 'Q.nonnegative(x - y - 2)'
assert str(prenex(ForAll(x, x > 0) & Exists(x, x < 0))) == "([('exists', x), ('forall', _x)], (_x > 0) & (x < 0))"
assert str(forall([x, z], x > y)) == 'ForAll(x, x > y)'
assert str(polynomial_abstraction(exp(x) > 1 + x, {x}).formula) == '_t > x + 1'
with assuming(x > 1):
    assert ask(Q.positive(x)) is True and len(global_assumptions) == 1
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_assumptions_leaves_do_not_load_sympy() -> None:
    """The leaves of :mod:`sympy_extras.assumptions` (facts, quantifiers,
    bounds, context) do not load SymPy."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_ASSUMPTIONS], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_ANALYSIS = r'''
import sys
from sympy_extras.core import symbols, Integer, Rational, pi, E, oo, sqrt
from sympy_extras.functions import exp, sin, log
from sympy_extras.sets import Interval, Reals
from sympy_extras.assumptions.facts import Facts
from sympy_extras.assumptions.intervals import (evaluate, interval_sign, signs_on_box, range_on_box, corner_signs,
    corner_value, monotone_range, isolate_signs)
from sympy_extras.assumptions.analysis import (certified_sign, certified_compare, domain_of, sign_on, box_of,
    signs_multivariate, decide_relational)
x, y = symbols('x y')
box = {x: (Integer(0), Integer(2)), y: (Integer(0), Integer(3))}
assert interval_sign(evaluate(exp(x) + y, box)) == 1
assert signs_on_box(x**2 + y**2 + 1, box) == frozenset([1])
assert str(range_on_box(x*y + 1, box)) == 'Interval(1, 7)'
assert corner_signs(exp(x) + y - 1, box, {x: (True, True), y: (True, True)}) == frozenset([1])
assert str(corner_value(exp(-x)*y + 1, {x: oo, y: Integer(2)})) == '1'
assert str(monotone_range(x + y, box, {x: (True, False), y: (False, False)})) == 'Interval.Lopen(0, 5)'
assert isolate_signs(sin(x), x, Interval(0, pi, True, True)) == frozenset([1])
assert isolate_signs(x*exp(x) - 1, x, Interval(0, 2, True, True)) == frozenset([-1, 0, 1])
assert certified_sign(E - 2) == 1 and certified_sign(sin(3) - Rational(1, 7)) == -1
assert certified_sign((1 + sqrt(2))**2 - 2*sqrt(2) - 3) == 0
assert certified_compare(pi, Integer(3)) == 1
facts = Facts([x > 0, x < 1, y >= 2])
assert str(domain_of(x, Facts([x > 0, x < pi]))) == 'Interval.open(0, pi)'
assert str(box_of([x, y], facts)) == '({x: (0, 1), y: (2, oo)}, {x: (True, True), y: (False, True)})'
assert signs_multivariate(x - y, [x, y], facts) == frozenset([-1])
assert sign_on(x**2 - 2, x, Interval(0, 10, True, True)) == frozenset([-1, 0, 1])
assert sign_on((x - 1)/(x**2 + 1), x, Reals) == frozenset([-1, 0, 1])
assert sign_on(1/x, x, Interval(-1, 1, True, True)) is None
assert decide_relational(x**2 > 1, Facts([x > 2])) is True
assert decide_relational(1/(x - 3) < 0, Facts([x > 0, x < 2])) is True
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_analysis_does_not_load_sympy() -> None:
    """:mod:`sympy_extras.assumptions.intervals` and
    :mod:`sympy_extras.assumptions.analysis` on rational functions,
    boxes and constants do not load SymPy (``continuous_domain`` and
    ``solveset`` of the other expressions and the CAD go through it)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_ANALYSIS], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_ASK = r'''
import sys
from sympy_extras.core import symbols, Q, Eq, sqrt
from sympy_extras.functions import Abs, sign, exp
from sympy_extras.sets import Integers, Reals
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.ask import ask
from sympy_extras.assumptions.refine import refine
from sympy_extras.assumptions.resolve import resolve
assert 'sympy' not in sys.modules
x, y = symbols('x y')
assert ask(x**3 > 0, x > 0) is True and ask(x*y > 0, (x > 0) & (y < 0)) is False
assert ask(Q.positive(x), Eq(x, 3)) is True and ask(x + y > 0, (x > 0) & (y > 0)) is True
assert ask(element(x + 1, Integers), element(x, Integers)) is True and ask(exp(x) > 0, element(x, Reals)) is True
assert str(refine(Abs(x) + sqrt(x**2), x > 0)) == '2*x' and str(refine(sign(x*y), (x > 0) & (y < 0))) == '-1'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_ask_refine_resolve_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.assumptions.ask`, ``refine`` and
    ``resolve`` does not load SymPy, nor do the questions answered by the
    predicates (the CAD, virtual substitution and the structure theorem go
    through SymPy at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_ASK], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_SOLVE = r'''
import sys
from sympy_extras.core import symbols
from sympy_extras.assumptions.solve import solve
assert 'sympy' not in sys.modules
x, a = symbols('x a')
assert str(solve(x**2 + 1, x)) == '{-I, I}' and str(solve(x**2 - 2, x)) == '{-sqrt(2), sqrt(2)}'
assert str(solve(a*x - 1, x, a > 0)) == '{1/a}'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_SAT = r'''
import sys
from sympy_extras.core import symbols
from sympy_extras.assumptions.sat import satisfiable, tautology, find_instance
assert 'sympy' not in sys.modules
x, y = symbols('x y')
assert satisfiable(True) == {} and satisfiable(False) is False and satisfiable((x > 1) & (x <= 1)) is False
assert tautology(True) is True and tautology((x > 0) | (x <= 0)) is True
assert str(find_instance(True, [x, y])) == '[{x: 0, y: 0}]'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_solve_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.assumptions.solve` does not load SymPy,
    nor do the rational equations over the complex numbers (``solveset``,
    the CAD and the triangular decompositions go through SymPy at run
    time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_SOLVE], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_sat_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.assumptions.sat` does not load SymPy,
    nor do the formulas decided without a SAT search or a decomposition
    (SymPy's SAT solver, the predicates' facts, the CAD and Hermite's
    count go through SymPy at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_SAT], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_STRUCTURE = r'''
import sys
from sympy_extras.core import symbols, sqrt, pi, I, Rational
from sympy_extras.functions import exp, log, sin, cos, atan
from sympy_extras.simplify.structure import ElementaryTower, canonical_form, is_zero, equal
assert 'sympy' not in sys.modules
x, y = symbols('x y')
assert is_zero(sin(x + y) - sin(x)*cos(y) - cos(x)*sin(y)) is True and is_zero(exp(x) - x - 1) is False
assert is_zero(log(x*y) - log(x) - log(y), [x > 0, y > 0]) is True
assert is_zero(pi/4 - 4*atan(Rational(1, 5)) + atan(Rational(1, 239))) is True
assert str(canonical_form((sqrt(x) + 1)**2 - x - 1)) == '2*sqrt(x)' and equal(exp(I*pi), -1) is True
tower = ElementaryTower()
tower.element(sqrt(x) + sqrt(y) + sqrt(x + y) + sqrt(2) + I)
assert tower.certified
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_structure_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.simplify.structure` does not load SymPy,
    nor do the identities which the tower decides without a sample point
    (the sample points of the region and the reliable values of the
    witnesses go through SymPy at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_STRUCTURE], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout
