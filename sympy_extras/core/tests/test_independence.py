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


_NATIVE_LIMITS = r'''
import sys
from sympy_extras.core import symbols, oo, sqrt, Integer
from sympy_extras.functions import exp, sin
from sympy_extras.assumptions.limits import limit, limit_seq, series
assert 'sympy' not in sys.modules
a, x = symbols('a x')
assert limit(sin(x)/x, x, Integer(0)) == 1 and limit(exp(a*x), x, oo, assumptions=a < 0) == 0
assert limit(x**a, x, oo, assumptions=a > 0) == oo
assert str(series(sqrt(a**2 + x), x, 0, 2, assumptions=a > 0)) == 'x/(2*a) + a + O(x**2)'
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


def test_native_limits_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.assumptions.limits` does not load
    SymPy, nor do the limits and series whose parameters have signs
    decided by the predicates (SymPy's ``limit_seq``, the reliable
    numerical values, and the SAT and CAD of the case distinctions go
    through SymPy at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_LIMITS], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_CAD = r'''
import sys
from fractions import Fraction
from sympy_extras.core import symbols
from sympy_extras.polys.cad.projection import projection_sets
from sympy_extras.polys.cad.samplepoints import SamplePoint, compare_real, real_roots_of
from sympy_extras.polys.cad.lifting import Lifting, cylindrical_algebraic_decomposition, lifting_for
assert 'sympy' not in sys.modules
x, y, z, w = symbols('x y z w')
cad = cylindrical_algebraic_decomposition([x**2 + y**2 + z**2 - 1, z - x*y], [x, y, z])
assert len(cad) == 137 and all(len(c.signs) == 2 for c in cad)
hong = cylindrical_algebraic_decomposition([x*w + y], [x, y, z, w])
assert hong.method == 'hong' and len(hong) == 21
projection = projection_sets([x**2 + y**2 - 1, x - y], [x, y], method='hong')
lifting = lifting_for(projection, [x, y], 'hong')
assert len(lifting.lift_all()[-1]) > 0
point = SamplePoint().extend(real_roots_of([1, 0, -2])[1]).extend(Fraction(1, 2))
assert point.sign(x**2 + y**2 - 2, [x, y]) == 1 and compare_real(real_roots_of([2, 0, -4])[1], point.values[0]) == 0
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_ISOLATION = r'''
import sys
from sympy_extras.core import symbols, Eq, Mod, oo
from sympy_extras.functions import exp, cos
from sympy_extras.sets import interval, finite_set
from sympy_extras.solvers.isolation import TranscendentalRoot, isolate_real_roots, point_between, compare_points
from sympy_extras.solvers.integers import (linear_diophantine_system, minimal_nonnegative_solutions, cooper,
    is_presburger, with_sides)
from sympy_extras.polys.roots import in_radicals, radical_form
assert 'sympy' not in sys.modules
x, y = symbols('x y')
assert str(isolate_real_roots(x**4 - 5*x**2 + 6, x)) == '[-sqrt(3), -sqrt(2), sqrt(2), sqrt(3)]'
assert str(isolate_real_roots((x**2 - 2)/(x + 3), x, interval(0, 5))) == '[sqrt(2)]'
r, s = TranscendentalRoot(exp(x) - x - 2, x, 1, 2), TranscendentalRoot(x - cos(x), x, 0, 1)
assert str(r.refined(4)) == 'TranscendentalRoot(-x + exp(x) - 2, x, 9/8, 19/16)'
assert str(r.evalf(20)) == '1.1461932206205825852'
assert compare_points(s, r) == -1 and str(point_between(s, r)) == '7/8' and point_between(r, oo) == 3
assert str(linear_diophantine_system([Eq(3*x + 5*y, 7)], [x, y])) == '([14 - 5*t0, 3*t0 - 7], [t0])'
assert minimal_nonnegative_solutions([[3, 5]], [22]) == ([[4, 2]], [])
assert str(cooper((x > y) & (x < y + 2), x)) == 'True' and is_presburger(Eq(Mod(x, 3), 1), {x})
assert str(with_sides(x - y > 0)) == 'x > y'
assert radical_form(x) is None and in_radicals(finite_set(1, 2)) == finite_set(1, 2)
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_cad_lifting_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.polys.cad.lifting` (with the
    projection and the sample points) does not load SymPy, nor do the
    decompositions, McCallum's and Hong's, and the sample points in
    algebraic number fields."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_CAD], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_isolation_integers_roots_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.solvers.isolation`,
    :mod:`sympy_extras.solvers.integers` and :mod:`sympy_extras.polys.roots`
    does not load SymPy, nor do the polynomial and rational roots, the root
    objects, the Diophantine systems and the eliminations which need no
    simplification (SymPy's ``real_roots`` with ``CRootOf``,
    ``continuous_domain``, ``solveset`` of the transcendental functions,
    ``simplify_logic`` and ``roots`` over ``EX`` go through it at run
    time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_ISOLATION], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_CAD_QE = r'''
import sys
from sympy_extras.core import symbols, Eq
from sympy_extras.polys.cad.qe import quantifier_elimination, decide, solution_set, sample_points, truth_tables
from sympy_extras.polys.cad.cylindrical import cylindrical_formula, cylindrical_set, cylindrical_cases
from sympy_extras.polys.cad.indexedroot import indexed_root
assert 'sympy' not in sys.modules
a, b, c, x, y, z = symbols('a b c x y z')
assert str(quantifier_elimination(Eq(a*x**2 + b*x + c, 0), [('exists', x)])) == \
    'Eq(c, 0) | (4*a*c - b**2 < 0) | (Ne(b, 0) & (4*a*c - b**2 <= 0))'
assert str(quantifier_elimination(Eq(z**2, x) & (z > y), [('exists', z)], free=[x, y])) == \
    '((x >= 0) & (y < 0)) | (x - y**2 > 0)'
assert decide(Eq(y, x**2), [('forall', x), ('exists', y)]) is True
assert str(solution_set(x**2 > 2, x)) == \
    'Union(Interval.open(-oo, IndexedRoot(x**2 - 2, x, 0)), Interval.open(IndexedRoot(x**2 - 2, x, 1), oo))'
assert str(sample_points((x**2 + y**2 < 1) & (x > y), [x, y])) == \
    '[{x: 0, y: -1/2}, {x: IndexedRoot(2*x**2 - 1, x, 1), y: 0}, {x: 3/4, y: 0}]'
assert truth_tables([x > 0, x**3 > 0], [x])[1] == [[False, False, True], [False, False, True]]
assert str(cylindrical_formula(x**2 + y**2 < 1, [x, y])) == '(x > -1) & (x < 1) & (y < sqrt(1 - x**2)) & (y > -sqrt(1 - x**2))'
assert 'IndexedRoot(x + y**3 - 3*y, y, 2)' in str(cylindrical_formula((y**3 - 3*y + x > 0) & (x > 1), [x, y]))
assert str(cylindrical_set((x**2 + y**2)*((x - 3)**2 + y**2 - 1) <= 0, [x, y])).endswith('{(0, 0)})')
assert str(cylindrical_cases(Eq(a*x, 1), [a], [x])) == '[((a > 0) | (a < 0), {1/a})]'
assert str(indexed_root(y**3 + x*y - 1, y, 0).subs(x, 1).evalf(20)) == '0.68232780382801932737'
from sympy_extras.sets import Reals
from sympy_extras.assumptions.sat import find_instance
from sympy_extras.assumptions.solve import solve
assert str(find_instance((x**2 + y**2 < 1) & (x > y), [x, y], domain=Reals)) == '[{x: 0, y: -1/2}]'
assert str(solve((x**2 + y**2 <= 1) & (x + y >= 1), [x, y], domain=Reals)) == \
    'ConditionSet((x, y), (x >= 0) & (x <= 1) & (y >= 1 - x) & (y <= sqrt(1 - x**2)), ProductSet(Reals, Reals))'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_IDEALS = r'''
import sys
from sympy_extras.core import Symbol, Integer, sqrt
from sympy_extras.polys.domains import QQ, GF
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.ideals import Ideal, separating_forms
from sympy_extras.polys.rings import WeightOrder, grevlex, lex
from fractions import Fraction
assert 'sympy' not in sys.modules
x, y, z, t, d = [Symbol(name) for name in 'xyztd']
I = Ideal([x*z - y**2, x**2 - y*z], x, y, z, domain=QQ)
assert (I.dimension(), I.degree()) == (1, 4) and str(I.eliminate([x])) == 'Ideal([y**4 - y*z**3], y, z)'
assert str(I.saturate(y)) == 'Ideal([x**2 - y*z, x*y - z**2, -x*z + y**2], x, y, z)'
assert str(I.hilbert_series(t)) == '(t**2 + 2*t + 1)/(1 - t)' and str(I.hilbert_polynomial(d)) == '4'
assert [str(g) for g in I.change_order(lex)] == ['x**2 - y*z', 'x*y**2 - y*z**2', 'x*z - y**2', 'y**4 - y*z**3']
assert len(I.change_order(WeightOrder((1, 2, 3), grevlex))) == 3
J = Ideal([x**2 - 2, y**2 - 3], x, y, domain=QQ)
assert str(J.minimal_polynomial(x + y, z).to_expr()) == 'z**4 - 10*z**2 + 1' and J.is_maximal() and J.is_prime()
assert J.radical() == J and [str(g) for g in J.groebner_basis(lex)] == ['x**2 - 2', 'y**2 - 3']
K = Ideal([x**3 + 2*x + 1, y**2 - x], x, y, domain=GF(7))
assert K.vector_space_dimension() == 6 and K.is_maximal()
A = Ideal([sqrt(Integer(2))*x - y, y**2 - 2], x, y,
          domain=AlgebraicField([Fraction(1), Fraction(0), Fraction(-2)], sqrt(Integer(2))))
assert A.univariate(x).to_expr() == x**2 - 1 and not A.is_maximal()
assert str(Ideal([x], x, y, domain=QQ).intersect(Ideal([y], x, y, domain=QQ))) == 'Ideal([x*y], x, y)'
assert str(Ideal([x**2*y, x*y**2], x, y, domain=QQ).quotient(x*y)) == 'Ideal([x, y], x, y)'
assert Ideal([x**2, y**3], x, y, domain=QQ).radical_contains(x*y)
assert str(list(separating_forms([x, y], 3))) == '[x, x + y, x - y, x + 2*y, x - 2*y]'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_VIRTUAL_SUBSTITUTION = r'''
import sys
from sympy_extras.core import symbols, Eq, Implies, Not
from sympy_extras.polys.virtual_substitution import (is_linear_in, is_quadratic_in, eliminate_linear,
    eliminate_quadratic, linear_quantifier_elimination, virtual_substitution_elimination, with_sides, to_nnf)
assert 'sympy' not in sys.modules
x, y, z = symbols('x y z')
assert is_linear_in((2*x + y > 0) & (x*y < 1), x) and not is_quadratic_in(x**3 > y, x)
assert str(with_sides((x - y > 0) & (x + 1 < 0))) == '(x > y) & (x < -1)'
assert str(to_nnf(Implies(x > 0, Not(y < 1)))) == '(y >= 1) | (x <= 0)'
assert str(eliminate_linear((x > y) & (x < 1), x)) == 'y < 1' and str(eliminate_linear(Eq(x, 2) & (x > y), x)) == 'y < 2'
assert str(eliminate_linear((x > 0) & (x < 0), x)) == 'False' and str(eliminate_quadratic(x**2 + 1 > 0, x)) == 'True'
assert str(linear_quantifier_elimination((x > y) & (y > z), [('exists', z), ('exists', x)])) == '(True, [])'
assert str(virtual_substitution_elimination(x**3 > y, [('exists', x)])) == "(x**3 > y, [('exists', x)])"
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_cad_qe_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.polys.cad.qe`,
    :mod:`sympy_extras.polys.cad.cylindrical` and
    :mod:`sympy_extras.polys.cad.indexedroot` does not load SymPy, nor do
    quantifier elimination, the decisions, solution sets, sample points,
    truth tables and cylindrical descriptions, nor the instances of
    ``find_instance`` and the systems of inequalities of ``solve`` which
    they answer."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_CAD_QE], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_virtual_substitution_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.polys.virtual_substitution` does not
    load SymPy, nor do the eliminations whose result is a truth value or
    a single relation (SymPy's ``simplify_logic`` of a compound result
    goes through it at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_VIRTUAL_SUBSTITUTION], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_ideals_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.polys.ideals` does not load SymPy, nor
    do the Gröbner bases (F4, FGLM, the walk), the eliminations, quotients,
    saturations, Hilbert series and the operations on zero-dimensional
    ideals, the modular Gröbner bases and Buchberger's algorithm."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_IDEALS], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout



_NATIVE_MODULAR_GROEBNER = r'''
import sys
from fractions import Fraction
from itertools import chain
from sympy_extras.core import Symbol
from sympy_extras.ntheory.generate import primerange
from sympy_extras.polys.buchberger import buchberger
from sympy_extras.polys.domains import QQ, ZZ, GF
from sympy_extras.polys.f4 import f4
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.modulargroebner import (ModularTrace, default_primes, direct_groebner, groebner,
    modular_groebner, rational_reconstruction)
from sympy_extras.polys.rings import PolyRing, WeightOrder, grevlex, lex
from sympy_extras.settings import configure
assert 'sympy' not in sys.modules
R = PolyRing(QQ, ['x', 'z', 'y'], lex)
x, z, y = R.gens
F = [x*(z + y.mul_ground(Fraction(210)) + R.one), x*(z + y.mul_ground(Fraction(420)) + R.from_int(2))]
trace = ModularTrace()
G = modular_groebner(F, R, primes=chain(primerange(2, 100), default_primes()), trace=trace)
assert [str(g) for g in G] == ['x*z', 'x*y + 1/210*x'] and trace.unlucky == [2, 3, 5, 7] and trace.lucky[0] == 11
assert modular_groebner(F, R) == f4(F, R) == buchberger(F, R) == groebner(F, R)
S = PolyRing(ZZ, ['x', 'y'], grevlex)
X, Y = S.gens
assert [str(g) for g in modular_groebner([(X*X).mul_ground(2) - Y, X*Y - S.one], S)] == ['2*x**2 - y', 'x*y - 1', 'y**2 - 2*x']
W = PolyRing(QQ, ['x', 'y'], WeightOrder((1, 3)))
assert modular_groebner([W.gens[0]**3 - W.gens[1], W.gens[1]**2 - W.one], W) == f4([W.gens[0]**3 - W.gens[1], W.gens[1]**2 - W.one], W)
assert rational_reconstruction(pow(3, -1, 101)*(-2) % 101, 101) == (-2, 3)
P = PolyRing(GF(7), ['x', 'y'], lex)
with configure(groebner_algorithm='buchberger'):
    assert direct_groebner([P.gens[0]**2 - P.gens[1]], P) == [P.gens[0]**2 - P.gens[1]]
u, v, w = [Symbol(name) for name in 'uvw']
for algorithm in ('f4', 'buchberger'):
    with configure(groebner_direct_time=0, groebner_algorithm=algorithm):
        I = Ideal([u**2 + v**2 + w**2 - 1, u*v - w/3, u + v - 2*w], u, v, w, domain=QQ)
        assert [str(g) for g in I.groebner_basis(lex)] == ['u + v - 2*w', 'v**2 - 2*v*w + 1/3*w', 'w**2 - 2/15*w - 1/5']
        assert I.vector_space_dimension() == 4
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''

_NATIVE_REGULAR_CHAINS = r'''
import sys
from sympy_extras.core import symbols
from sympy_extras.polys.regularchains import RegularChain, triangularize, regular_gcd
from sympy_extras.solvers.parametric import parametric_cases
assert 'sympy' not in sys.modules
a, b, c, p, q, x, y, z = symbols('a b c p q x y z')
assert str(triangularize([x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1], x, y, z)[0].polys) == \
    '[z**2 + 2*z - 1, y - z, x - z]'
f = x**3 + p*x + q
assert str(triangularize([f, f.diff(x)], x, p, q, mode='kalkbrener')) == \
    '[RegularChain([4*p**3 + 27*q**2, 2*p*x + 3*q], x, p, q)]'
assert str(triangularize([x**2 - y**2, x*(y - 1)], x, y, inequations=[x])) == \
    '[RegularChain([y - 1, x + 1], x, y), RegularChain([y - 1, x - 1], x, y)]'
T = RegularChain([z**2 - 2, y**2 - z, (y + 1)*x - 1], x, y, z)
assert T.contains(y**4 - 2) and str(T.reduce(y**3 + x)) == 'y*z + 3' and T.is_regular(y - z)
assert str(RegularChain([y**2 - 1, x**2 - y], x, y).regularize(y - 1)[0]) == '[RegularChain([y - 1, x**2 - y], x, y)]'
assert str(regular_gcd(x**2 - y, x**2 - 3*x + 2*y, x, RegularChain([y**2 - 1], x, y))) == \
    '[(1, RegularChain([y + 1], x, y)), (x - 1, RegularChain([y - 1], x, y))]'
assert str(RegularChain([z*y - 1, y*x - z], x, y, z).saturated_ideal().reduced().exprs) == '[x*y - z, y*z - 1, -x + z**2]'
[chain] = triangularize([x**2 + y**2 - 5, x*y - 1], x, y)
assert str(chain.numerical_solutions(5)[0]) == '{y: -2.1889, x: -0.45685}'
assert str(parametric_cases([a*x**2 + b*x + c], [x])[1]) == 'ParametricCase(Eq(a, 0) & Ne(b, 0), {(-c/b,)})'
assert str(parametric_cases([a*x + y - 1, x + a*y - 1], [x, y])) == \
    '[ParametricCase(Eq(a, 1), {(1 - y, y)}), ParametricCase(Ne(a + 1, 0), {(1/(a + 1), 1/(a + 1))})]'
from sympy_extras.assumptions.solve import solve
assert str(solve([a*x - 1], [x], cases=True)) == 'ConditionSet(x, Ne(a, 0), {1/a})'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_HERMITE_COMPREHENSIVE = r'''
import sys
from sympy_extras.core import symbols, Eq, Ne
from sympy_extras.polys.hermite import count_real_solutions, count_complex_solutions, sign_determination, \
    real_sign_counts, tarski_query, hermite_matrix, decide_zero_dimensional
from sympy_extras.polys.comprehensive import comprehensive_groebner_system, exists_complex, \
    complex_quantifier_elimination, reduce_complex
assert 'sympy' not in sys.modules
a, b, x, y = symbols('a b x y')
assert count_real_solutions([x**2 + y**2 - 1, x - y], x, y) == 2 and count_complex_solutions([x**2 + 1, (y - 1)**2], x, y) == 2
assert str(sign_determination([x**2 - 1, y**2 - 1], [x + y, x], x, y)) == '{(-1, -1): 1, (0, -1): 1, (0, 1): 1, (1, 1): 1}'
assert str(real_sign_counts([x**3 - x, y - x**2], x - y, x, y)) == 'SignCounts(zero=2, positive=0, negative=1)'
assert tarski_query([x**3 - x, y - x**2], 2*y - 1, x, y) == 1 and str(hermite_matrix([x**2 - 2], x, x)) == 'ExprMatrix([[0, 4], [4, 0]])'
assert decide_zero_dimensional(Eq(x**2 + y**2, 4) & Eq(x*y, 1) & (x > y), [x, y]) is True
assert [repr(r) for r in comprehensive_groebner_system([a*x - b], [x], [a, b])] == \
    ['Branch([], [a], [a*x - b])', 'Branch([a], [b], [1])', 'Branch([a, b], [], [])']
assert str(exists_complex([a*x - b], [], [x])) == 'Ne(a, 0) | (Eq(a, 0) & Eq(b, 0))'
assert str(complex_quantifier_elimination(Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0), [('exists', x)])) == 'Eq(a**2 - 4*b, 0)'
assert str(reduce_complex(Eq(x**2 - y, 0) & Eq(x*y - 1, 0) & Ne(x, 0))) == 'Eq(x*y - 1, 0) & Eq(-x + y**2, 0) & Eq(x**2 - y, 0)'
from sympy_extras.sets.fancysets import Complexes
from sympy_extras.assumptions.quantifiers import Exists, ForAll
from sympy_extras.assumptions.resolve import resolve
from sympy_extras.assumptions.sat import _without_real_solution
assert str(resolve(Exists(x, Eq(a*x, 1)), domain=Complexes)) == 'Ne(a, 0)'
assert str(resolve(ForAll(x, Ne(x**2, a)), domain=Complexes)) == 'False'
assert str(resolve(Exists(x, Eq(a*x, b)), domain=Complexes, assumptions=Ne(a, 0))) == 'True'
assert _without_real_solution(Eq(x**2 + y**2, 1) & Eq(x*y, 1) & (x > 0), [x, y])
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''



def test_native_modular_groebner_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.polys.modulargroebner` and
    :mod:`sympy_extras.polys.buchberger` does not load SymPy, nor do the
    modular Gröbner bases over ``QQ`` and ``ZZ`` (with unlucky primes, a
    weight order), Buchberger's algorithm, the dispatcher, and the Gröbner
    bases of :class:`~sympy_extras.polys.ideals.Ideal` by the modular
    algorithm and by Buchberger's algorithm."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_MODULAR_GROEBNER], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout

def test_native_regular_chains_and_parametric_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.polys.regularchains` and
    :mod:`sympy_extras.solvers.parametric` does not load SymPy, nor do the
    triangular decompositions, the operations on chains, their saturated
    ideals and numerical solutions, the cases of the parameters and the
    ``solve(..., cases=True)`` which they answer (the exact solutions of a
    chain, SymPy's ``CRootOf``, go through SymPy at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_REGULAR_CHAINS], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_hermite_comprehensive_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.polys.hermite` and
    :mod:`sympy_extras.polys.comprehensive` does not load SymPy, nor do the
    counts of real solutions, the sign determination, the decisions of
    zero-dimensional systems (also the one of ``satisfiable``), the
    comprehensive Gröbner systems, the eliminations over the complex
    numbers and ``resolve`` with ``domain=Complexes``."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_HERMITE_COMPREHENSIVE], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_DECOMPOSITION_RUR = r"""
import sys
from sympy_extras.core import symbols
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.idealdecomposition import (radical, equidimensional_parts, minimal_primes,
    primary_decomposition, associated_primes, is_primary, independent_set)
from sympy_extras.polys.rur import rur
assert 'sympy' not in sys.modules
t, x, y, z = symbols('t x y z')
assert str(radical(Ideal([x**2*y, x*y**2], x, y, domain=QQ))) == 'Ideal([x*y], x, y)'
assert str(Ideal([x**2*y, x*y**3], x, y, domain=QQ).radical()) == 'Ideal([x*y], x, y)'
assert str(minimal_primes(Ideal([x*z - y**2, x**2 - y*z], x, y, z, domain=QQ))) == \
    '[Ideal([x - z, y - z], x, y, z), Ideal([x, y], x, y, z), Ideal([y**2 + y*z + z**2, x + y + z], x, y, z)]'
assert str(equidimensional_parts(Ideal([x*z, x*y, x*(x - 1)*(z - 1)], x, y, z, domain=QQ))) == \
    '[Ideal([x], x, y, z), Ideal([x - 1, y, z], x, y, z)]'
assert str(primary_decomposition(Ideal([x**2, x*y], x, y, domain=QQ))) == \
    '[(Ideal([x], x, y), Ideal([x], x, y)), (Ideal([x**2, y], x, y), Ideal([x, y], x, y))]'
# an irrational embedded point and a point of dimension zero
f = x**2 - 2
pair, point = Ideal([f, y], x, y, domain=QQ), Ideal([x - 1, y - 1], x, y, domain=QQ)
assert len(primary_decomposition((pair**2).intersect(point))) == 2
assert str(associated_primes(Ideal([x*y, x*z, y*z], x, y, z, domain=QQ))) == \
    '[Ideal([x, y], x, y, z), Ideal([x, z], x, y, z), Ideal([y, z], x, y, z)]'
assert is_primary(Ideal([x**2, y**3], x, y, domain=QQ)) and not Ideal([x**2, x*y], x, y, domain=QQ).is_primary()
assert str(independent_set(Ideal([x*y, x*z], x, y, z, domain=QQ))) == '[y, z]'
assert Ideal([x**2 - y**3], x, y, domain=QQ).is_prime()
R = rur([x**2 + y**2 - 1, x*y], x, y, t=t)
assert str(R.form) == 'x + 2*y' and str(R.polynomial.to_expr()) == 't**4 - 5*t**2 + 4'
assert str([g.to_expr() for g in R.polynomials()]) == '[-t**3/3 + 4*t/3, t**3/6 - t/6]'
assert str(R.numerical_solutions(5)[1]) == '{x: -1.0000, y: 0.0}'
S = Ideal([(x - 1)**2*(x + 2), y - x**2], x, y, domain=QQ).rur(t)
assert S.multiplicity(1) == 2 and S.multiplicity(-2) == 1 and S.multiplicity(5) == 0
assert str(rur([x**2 - 2, y - x - 1], x, y, t=t).coordinates(2)) == '[1, 2]'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
"""


def test_native_decomposition_and_rur_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.polys.idealdecomposition` and
    :mod:`sympy_extras.polys.rur` does not load SymPy, nor do the radicals,
    equidimensional parts, minimal and associated primes, primary
    decompositions and the methods of the native ``Ideal`` which call them,
    nor the rational univariate representations, their coordinate
    polynomials, numerical solutions, multiplicities and coordinates (the
    exact solutions, SymPy's ``CRootOf``, go through SymPy at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_DECOMPOSITION_RUR], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_DIFFERENTIAL = r'''
import sys
from sympy_extras.core import Eq, Function, symbols
from sympy_extras.polys.differential import (DifferentialPolynomial, DifferentialRing, janet_basis,
    janet_multiplicative_variables, rosenfeld_groebner)
assert 'sympy' not in sys.modules
x, y, z, t, g = symbols('x y z t g')
u = Function('u')(x, y, z)
J = janet_basis([u.diff(z, 2) + y*u.diff(x, 2), u.diff(y, 2)], [u])
assert len(J.equations) == 7 and J.dimension == 12 and J.contains(u.diff(z, 4)) and not J.contains(u.diff(z, 3))
assert str(J.hilbert_series(t)) == 't**4 + 3*t**3 + 4*t**2 + 3*t + 1' and J.hilbert_polynomial(t) == 0
v = Function('v')(x, y)
E = janet_basis([v.diff(x, 2) - y**2*v, Eq(v.diff(y), x*v)], [v])
assert str(E.reduce(v.diff(x, y))) == '(x*y + 1)*v(x, y)' and str(E.series_solution(2)[v]) == 'C0*x*y + C0'
assert janet_multiplicative_variables([(2, 1), (1, 2), (0, 3)]) == [(True, True), (False, True), (False, True)]
p = Function('p')(x)
I = rosenfeld_groebner([p.diff(x)**2 - 4*p], [p])
assert str([(c.equations, c.inequations) for c in I.components]) == \
    '[([-4*p(x) + Derivative(p(x), x)**2], [Derivative(p(x), x)]), ([p(x)], [])]'
assert I.contains(p.diff(x)*(p.diff(x, 2) - 2)) and not I.contains(p.diff(x, 2) - 2)
assert [len(c.regular_chains()) for c in I.components] == [1, 1] and I.components[0].saturation()
X, Y, L = Function('X')(t), Function('Y')(t), Function('L')(t)
P = rosenfeld_groebner([X.diff(t, 2) + L*X, Y.diff(t, 2) + L*Y + g, X**2 + Y**2 - 1], [L, X, Y])
assert len(P.components) == 3 and P.components[0].contains(L.diff(t) + 3*g*Y.diff(t))
R = DifferentialRing([v])
q = DifferentialPolynomial.from_expr(R, (x*v - 1)*v.diff(x)**3 + v*v.diff(x))
assert q.degree() == 3 and str(q.initial().to_expr()) == 'x*v(x, y) - 1'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_differential_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.polys.differential` does not load
    SymPy, nor do the Janet bases (normal forms, Hilbert series, series
    solutions), the Rosenfeld-Groebner algorithm (factors, saturations,
    regular chains, membership) and Ritt's reduction."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_DIFFERENTIAL], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_PUISEUX_FUNCTIONFIELDS = r"""
import sys
from fractions import Fraction
from sympy_extras.core import symbols, Integer, I, oo, sqrt
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.puiseux import (puiseux_expansions, adjoin_root, field_from_polynomial, field_of,
    minimal_polynomial_of, in_subfield)
from sympy_extras.polys.functionfields import function_field, integral_basis
from sympy_extras.polys.cad.indexedroot import IndexedRoot, bound_variable
assert 'sympy' not in sys.modules
x, y, a = symbols('x y a')
assert str([b.as_expr() for b in puiseux_expansions(y**2 - x**3 - x**2, x, y, Integer(0), 4)]) == \
    '[x**3/8 - x**2/2 - x, -x**3/8 + x**2/2 + x]'
assert str([b.as_expr() for b in puiseux_expansions(y**2 - x**3 - 1, x, y, oo, 3)]) == '[x**(3/2) + 1/(2*x**(3/2))]'
[b] = puiseux_expansions(y**2 - 2 - x, x, y, Integer(0), 2)
assert (b.conjugates, str(b.field)) == (2, 'QQ<sqrt(2)>')
assert sum(b.ramification * b.conjugates for b in puiseux_expansions(y**2 - x**2 - 1, x, y, I, 3)) == 2
t = bound_variable('t')
branches = puiseux_expansions(y**4 - 4*x*y**2 + 4, x, y, IndexedRoot(t**3 - t - 1, t, 0), 2)
assert sum(b.ramification * b.conjugates for b in branches) == 4
K = field_from_polynomial([Fraction(1), Fraction(0), Fraction(-2)])
L, into, u = adjoin_root(K, [K.one, K.zero, K.neg(K.alpha)])
assert minimal_polynomial_of(u, L) == [1, 0, 0, 0, -2] and in_subfield(u, L, into) is None
assert field_of(sqrt(Integer(3)))[0].degree == 2
E = function_field(y**4 + (x**2 + x)*y - x**2, x, y, QQ)
assert str([w.as_expr() for w in E.integral_basis()]) == '[1, y, y**2, y**3/x]'
assert str([w.as_expr() for w in E.normal_basis()]) == '[1, y, y**2, y**3/x]'
assert E.pole_orders_at_infinity(E.normal_basis()) == [0, 1, 2, 1]
assert str([w.as_expr() for w in E.local_basis_at_infinity()]) == '[1, y/x, y**2/x**2, y**3/x**2]'
assert E.is_integral(E.element(y**3/x)) and not E.is_integral(E.element(y/x))
assert str(integral_basis(y**2 - (x - a)**2*(x - 1), x, y)) == '[1, y/(-a + x)]'
F = function_field(y**2 - 2*x**2 - x**3, x, y, K)
assert str([w.as_expr() for w in F.integral_basis()]) == '[1, y/x]'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
"""


def test_native_puiseux_and_function_fields_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.polys.puiseux` and
    :mod:`sympy_extras.polys.functionfields` does not load SymPy, nor do
    the expansions at rational, algebraic (radicals, ``I``, real roots) and
    infinite points, the exact number fields, the integral bases over the
    rationals, a number field and the parameters, the local and normal
    bases at infinity and the integrality test."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_PUISEUX_FUNCTIONFIELDS], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_THUE = r'''
import sys
from sympy_extras.core import symbols
from sympy_extras.solvers.thue_equation import ThueEquation, elements_of_norm, thue, units_of_order
assert 'sympy' not in sys.modules
x, y = symbols('x y')
assert thue(x**3 - 2*y**3, 1, x, y) == [(-1, -1), (1, 0)]
assert thue(x**3 + x**2*y - 2*x*y**2 - y**3, 1, x, y) == [(-9, 5), (-1, -1), (-1, 1), (-1, 2), (0, -1), (1, 0),
                                                          (2, -1), (4, -9), (5, 4)]
assert thue(x**4 + y**4, 2, x, y) == [(-1, -1), (-1, 1), (1, -1), (1, 1)]
try:
    thue(x**3 - y**3, 1, x, y)
except ValueError:
    pass
else:
    raise AssertionError('a reducible form should be refused')
units, torsion = units_of_order([1, 0, 0, -2])
assert units in ([(1, -1, 0)], [(1, 1, 1)]) and elements_of_norm([1, 0, 0, -2], 2, units) == [(0, 1, 0)]
assert ThueEquation([1, 0, 0, -2], 1).small_solutions(2) == [(-1, -1), (1, 0)]
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_NUMERIC = r'''
import random, sys
from sympy_extras.core import symbols, sqrt, Integer, Rational, I, pi
from sympy_extras.functions import Log, log, sin, cos, exp, atan
from sympy_extras._numeric import reliable_form, reliable_value, reliable_complex
x = symbols('x')
z = -6 + (-2 + sqrt(2))**2 + 4*sqrt(2)
assert str(reliable_form(Log(-2*sqrt(2) - 2*sqrt(z)))) == 'log(2*sqrt(2)) + I*pi'
assert reliable_form(sqrt(sin(1)**2 + cos(1)**2 - 1) + 1) is None
assert str(reliable_value(Log(-x - sqrt(z)), 15, {x: Integer(2)})) == '0.693147180559945 + 3.14159265358979*I'
assert reliable_complex(sqrt(x), 15, {x: Integer(-4)}) == 2j
assert str(reliable_value(exp(-1000000*x**2 - 50), 30, {x: Integer(2)})) == '2.27857274099872593085040751845e-1737200'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_thue_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.solvers.thue_equation` does not load
    SymPy, nor do the resolution of Thue equations (the units of the
    order, the elements of a given norm, Baker's bound, de Weger's
    reduction and the enumeration), on totally complex forms too, and the
    refusal of reducible forms."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_THUE], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_numeric_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras._numeric` does not load SymPy, nor do
    the hidden zeros (proved by the minimal polynomial), the cuts and the
    values at points."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_NUMERIC], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_POLYS = r"""
import importlib
import pkgutil
import sys
import sympy_extras.polys as polys
modules = []
packages = [(list(polys.__path__), 'sympy_extras.polys.')]
while packages:
    path, prefix = packages.pop()
    for info in pkgutil.iter_modules(path, prefix):
        last = info.name.rsplit('.', 1)[-1]
        if last in ('tests', 'interop') or last.endswith('_sympy'):
            continue
        module = importlib.import_module(info.name)
        modules.append(info.name)
        if info.ispkg:
            packages.append((list(module.__path__), info.name + '.'))
assert len(modules) > 70, modules
assert 'sympy' not in sys.modules, [m for m in modules if 'sympy' in sys.modules]
from fractions import Fraction
from sympy_extras.core import Integer, Rational, symbols
from sympy_extras.polys.domains import GF, QQ, ZZ
from sympy_extras.polys.f4 import f4
from sympy_extras.polys.groebnerwalk import extended_groebner, groebner_walk
from sympy_extras.polys.hilbert import hilbert_numerator, hilbert_series
from sympy_extras.polys.orderings import WeightOrder, elimination_order, grevlex, lex
from sympy_extras.polys.polyerrors import NotInvertible
from sympy_extras.polys.polytools import cancel, factor, invert
from sympy_extras.polys.rings import PolyRing
from sympy_extras.polys.subresultants import dup_psc, psc
x, y, z, t = symbols('x y z t')
R = PolyRing(QQ, ['x', 'y', 'z'], grevlex)
X, Y, Z = R.gens
G = f4([X*Z - Y**2, X**2 - Y*Z], R)
for lift in ('normal_form', 'cofactors'):
    assert [str(g) for g in groebner_walk(G, R, lex, check=True, lift=lift)] == \
        ['x**2 - y*z', 'x*y**2 - y*z**2', 'x*z - y**2', 'y**4 - y*z**3']
assert [str(g) for g in groebner_walk(G, R, WeightOrder((1, 2, 3), lex))] == ['y**3 - x**3', 'y*z - x**2', 'x*z - y**2']
S = PolyRing(GF(7), ['x', 'y'], elimination_order(1, 2))
H, C = extended_groebner([S.gens[0]**2 - S.gens[1], S.gens[0]*S.gens[1] - S.one], S)
assert len(H) == len(C)
assert hilbert_numerator([(2, 0), (0, 2)], t).rep == [1, 0, -2, 0, 1]
assert str(hilbert_series([(2, 0, 0), (1, 1, 0), (0, 2, 0)], 3, t)) == '(2*t + 1)/(1 - t)'
assert dup_psc([1, 0, 1], [1, 0, -1], ZZ) == [4, 0, 1]
T2 = PolyRing(ZZ, ['x', 'y'])
assert [str(c) for c in psc(T2.gens[0]**2*T2.gens[1] + T2.gens[0], T2.gens[0] + T2.gens[1])] == ['y**3 - y', '1']
assert str(invert(x**2 - 1, 2*x - 1)) == '-4/3' and invert(Integer(3), Integer(7)) == Integer(5)
for a, b, error in [(x*y + 1, y**2 + 1, ValueError), (Rational(1, 2), Integer(3), NotInvertible)]:
    try:
        invert(a, b)
    except error:
        pass
    else:
        raise AssertionError('%s expected' % error.__name__)
assert str(cancel((x**2 - 1)/(x - 1))) == 'x + 1' and str(cancel((-x*y - y)/(x*y + x), x)) == '(-x*y - y)/(x*(y + 1))'
assert str(factor(x**2*y - y**3)) == 'y*(x - y)*(x + y)'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
"""


def test_native_polys_do_not_load_sympy() -> None:
    """Every module of :mod:`sympy_extras.polys` but ``interop`` and the
    SymPy faces (``*_sympy``) imports without SymPy, and the Gröbner walk,
    the orders, the Hilbert series, the principal subresultant
    coefficients and ``invert``, ``cancel`` and ``factor`` over the ported
    domains run without it."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_POLYS], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_KOVACIC = r'''
import sys
from sympy_extras.core import Eq, Function, Rational, symbols
from sympy_extras.functions import exp
from sympy_extras.solvers.kovacic import KovacicSolution, dsolve_kovacic, liouvillian_solution, normal_form
from sympy_extras.solvers._special_values import (as_zero, determinant, explicit_solution, general_solution_cases,
    ode_order, solution_jacobian, defined_equation)
assert 'sympy' not in sys.modules
x, a, b, C1, C2 = symbols('x a b C1 C2')
y = Function('y')(x)
assert normal_form(1/x, 1 - 1/(4*x**2), x) == -1
assert liouvillian_solution(x, x) is None and liouvillian_solution(x**3, x) is None
assert liouvillian_solution(1/x**3, x) is None
z = liouvillian_solution(Rational(0), x)
assert isinstance(z, KovacicSolution) and z.case == 1 and z.solution == 1
assert dsolve_kovacic(y.diff(x, 2) - x*y, y) is None and dsolve_kovacic(y.diff(x, 2) - y**2, y) is None
assert ode_order(y.diff(x, 2) + y.diff(x)**3, y) == 2
assert str(solution_jacobian(C1*exp(a*x) + C2*exp(b*x), [C1, C2], x)) == '-a*exp(a*x)*exp(b*x) + b*exp(a*x)*exp(b*x)'
assert determinant([[a, b, 0, 0], [0, a, b, 0], [0, 0, a, b], [b, 0, 0, a]]).expand() == a**4 - b**4
assert explicit_solution([Eq(y, C1*exp(x))], y) == C1*exp(x) and as_zero(Eq(y, x)) == y - x
assert general_solution_cases(C1*exp(x), y.diff(x) - y, y, lambda e: None) == C1*exp(x)
assert not defined_equation((a**(b*x)).subs(a, 0), x)
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_CONDITIONS = r'''
import random, sys
from sympy_extras.core import symbols, sqrt, Integer, Rational, I, pi, Eq
from sympy_extras.core.logic import or_
from sympy_extras.functions import Abs, arg, log, sin, cos, exp, piecewise, gamma
from sympy_extras.sets import Reals
from sympy_extras.assumptions.facts import element
from sympy_extras.integrals.conditions import (ConditionalValue, decide, real_form, canonical, positive,
    sample_values, numerically_equal, numerical_verdict, combine)
a, b, x, k = symbols('a b x k')
v = ConditionalValue(1/a, a > 0).add(ConditionalValue(a, a < 1))
assert str(v) == 'ConditionalValue(a + 1/a, (a > 0) & (a < 1))'
assert str(ConditionalValue(piecewise((a, a > 1), (-a, a < 0)), or_(a > 1, a < 0)).as_piecewise(Integer(0))) == \
    'Piecewise((a, a > 1), (-a, a < 0), (0, True))'
assert str(decide((a > 0) & (b > 1), a > 1)) == 'b > 1' and decide(a < 0, a > 1) is None
assert str(real_form(Abs(arg(a)) < pi/2, element(a, Reals))) == 'a > 0' and str(canonical(k + 1 < 1)) == 'k < 0'
assert positive(a*Abs(b), a > 0)
point = sample_values([a, b], [a > 1, b < a], random.Random(1))
assert point is not None and point[a] > 1 and point[b] < point[a]
assert numerically_equal(sin(x)**2 + cos(x)**2, Integer(1)) and not numerically_equal(sqrt(x**2), x, x < 0)
assert not numerically_equal(exp(-10**6*x**2), Integer(0), x > 1)     # the structure theorem's second opinion
assert numerical_verdict(log(-a), log(a) + I*pi, None, a > 0) is True
assert numerical_verdict(gamma(Integer(0)), gamma(Integer(0)) + 1) is None
assert str(combine([ConditionalValue(a, a > 0), ConditionalValue(1, a < 2)])) == 'ConditionalValue(a + 1, (a > 0) & (a < 2))'
from sympy_extras.assumptions.ask import ask
from sympy_extras.functions import atan
assert ask(Eq(4*atan(Rational(1, 5)) - atan(Rational(1, 239)), pi/4)) is True
assert ask(Eq(log(x**2), 2*log(x)), x > 0) is True and ask(Eq(log(x**2), 2*log(x))) is None
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_CHARPIT_DAE = r'''
import sys
from sympy_extras.core import Eq, Function, symbols
from sympy_extras.functions import cos, sin
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.solvers.charpit import check_complete_integral, complete_integral
from sympy_extras.solvers.dae import (check_dae, core_nilpotent_decomposition, dae_index, dae_matrices, dsolve_dae,
    matrix_exponential)
assert 'sympy' not in sys.modules
x, y, a, b = symbols('x y a b')
u = Function('u')(x, y)
p, q = u.diff(x), u.diff(y)
for equation in (p*q - 1, Eq(u, p*x + q*y + p*q), p**2 + q**2 - 1, p**2 - q, p*x - q*y, p*q - x):
    solution = complete_integral(equation, u)
    assert solution is not None and check_complete_integral(equation, u, solution), (equation, solution)
assert str(complete_integral(p*q - 1, u)) == 'Eq(u(x, y), a*x + b + y/a)'
assert complete_integral(p*q - u*x*y, u) is None
assert not check_complete_integral(p*q - 1, u, Eq(u, a*x + y + b))
E = ExprMatrix.from_rows
T, C, N, k = core_nilpotent_decomposition(E([[2, 0, 0], [0, 0, 1], [0, 0, 0]]))
assert (C, N, k) == (E([[2]]), E([[0, 1], [0, 0]]), 2)
assert dae_index(E([[1, 0], [0, 0]]), E([[0, 1], [1, 0]])) == 2
A, B, f = E([[1, 0], [0, 0]]), E([[0, -1], [1, 0]]), ExprMatrix.column([0, sin(x)])
result = dsolve_dae(A, B, f, x)
assert result.solution == ExprMatrix.column([sin(x), cos(x)]) and result.index == 2
oscillator = dsolve_dae(ExprMatrix.eye(2), B, None, x)
assert len(oscillator.constants) == 2 and check_dae(ExprMatrix.eye(2), B, None, oscillator.solution, x)
jordan = dsolve_dae(ExprMatrix.eye(2), E([[1, 1], [0, 1]]), None, x)
assert check_dae(ExprMatrix.eye(2), E([[1, 1], [0, 1]]), None, jordan.solution, x)
y1, y2 = Function('y1')(x), Function('y2')(x)
assert dae_matrices([Eq(y1.diff(x), y2), Eq(y1, sin(x))], [y1, y2])[:3] == (A, B, f)
assert str(matrix_exponential(E([[1, 1], [0, 1]]), x)) == 'ExprMatrix([[exp(x), x*exp(x)], [0, exp(x)]])'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_kovacic_and_ode_special_values_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.solvers.kovacic` and
    :mod:`sympy_extras.solvers._special_values` does not load SymPy, nor
    do the normal form, the decision that an equation has no Liouvillian
    solution (the poles, the sets of the cases and their linear systems),
    the order of an equation, the Jacobians of the solutions and an
    equation without parameters (SymPy's ``integrate`` and its Riccati
    solver go through SymPy at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_KOVACIC], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_charpit_and_dae_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.solvers.charpit` and
    :mod:`sympy_extras.solvers.dae` does not load SymPy, nor do the
    complete integrals whose equations are polynomial in the unknowns and
    whose integrands are sums of powers (Charpit's standard forms, Clairaut,
    the separable and first integral cases, the checks), the
    core-nilpotent decomposition, the index, the solutions of
    differential-algebraic systems whose regular part needs no integral
    (Putzer's matrix exponential with complex and repeated eigenvalues,
    the nilpotent part) and the extraction of the matrices."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_CHARPIT_DAE], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_conditions_do_not_load_sympy() -> None:
    """The package ``sympy_extras.integrals`` exports nothing, so importing
    :mod:`sympy_extras.integrals.conditions` does not load the integrators
    written on SymPy; its decisions, sample points and numerical
    comparisons (with the structure theorem as a second opinion), and the
    numerical step of the native ``ask``, run without SymPy."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_CONDITIONS], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_LIE_PDE = r'''
import sys
from sympy_extras.core import Function, symbols
from sympy_extras.functions import exp, sin
from sympy_extras.solvers.lie import JetSpace, Symmetry, symmetries, check_symmetry, symmetry_janet_basis
from sympy_extras.solvers.pde import pde_symmetries, similarity_reduction, check_pde_solution
x, t, m = symbols('x t m')
u = Function('u')(x, t)
y = Function('y')(x)
burgers = u.diff(t) + u*u.diff(x) - u.diff(x, 2)
syms = symmetries(burgers, u)
assert len(syms) == 5 and all(check_symmetry(burgers, u, s) for s in syms)
assert str(symmetry_janet_basis(u.diff(t) + u*u.diff(x) + u.diff(x, 3), u).dimension) == '4'
assert len(symmetries(u.diff(t) - (u**m*u.diff(x)).diff(x), u)) == 4
assert len(pde_symmetries(u.diff(t) - u.diff(x, 2), u, degree=3)) == 6
assert len(symmetries(y.diff(x) + 2*x*y - x*exp(-x**2), y, basis=[exp(-x**2)])) >= 1
assert len(symmetries(u.diff(x, 2)**2 - u.diff(t), u, degree=1)) == 6
r = similarity_reduction(burgers, u, Symmetry(JetSpace([u]), [0, 1], [0]))
assert str(r.z) == 'x', r
assert check_pde_solution(u.diff(t) - u.diff(x, 2), u, exp(-t)*sin(x))
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_MELLIN = r'''
import sys
from sympy_extras.core import Integer, S, oo, sqrt, symbols
from sympy_extras.assumptions.facts import element
from sympy_extras.functions import Abs, besselj, besselk, erfc, exp, hyper, log, sin
from sympy_extras.sets import Reals
from sympy_extras.integrals.mellin import (KERNELS, GammaQuotient, decompose_integrand, mellin_kernel,
    mellin_transform, monomial, poles_separated)
x, s, a, b = symbols('x s a b')
assert str(mellin_transform(exp(-2*x), x, s)) == 'MellinTransform(gamma(s)/2**s, (0, oo))'
assert str(mellin_transform(sin(x)/x, x, s)) == \
    'MellinTransform(2**s*sqrt(pi)*gamma(s/2)/(4*gamma(3/2 - s/2)), (0, 2))'
assert str(mellin_transform(exp(-x)*log(x)**2, x, s).transform) == '(polygamma(0, s)**2 + polygamma(1, s))*gamma(s)'
assert str(mellin_transform(hyper([a], [b], -x), x, s).transform) == \
    'gamma(b)*gamma(s)*gamma(a - s)/(gamma(a)*gamma(b - s))'
assert mellin_transform(besselj(Integer(0), 3*x)/(x**2 + 4), x, s) is None
assert monomial(3*a*sqrt(x), x) == (3*a, S.Half)
assert str(mellin_kernel(1/(x + 2)**a, x)) == 'Match((1 + x)**(-a), beta=1/2, gamma=1, constant=2**(-a))'
p = decompose_integrand(x*besselk(Integer(0), x)**2, x)
assert [m.kernel.name for m in p.matches] == ['besselk*besselk']
assert [m.kernel.name for m in decompose_integrand(erfc(x), x, 'lower').matches] == ['erfc', 'theta(1 - x)']
q = mellin_kernel(sin(a*x), x).quotient(element(a, Reals))
assert q.bases[-1][0] == Abs(a)
q = GammaQuotient(1, [(2, -1)], [(0, S.One)], [], 0, oo).compose(1, -2)
assert str(q.as_expr(s)) == '2**(2*s)*gamma(1 - 2*s)/2' and q.strip == (-oo, S.Half)
assert all(poles_separated(k.quotient) for k in KERNELS.values())
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_TRANSCENDENTAL = r'''
import sys
from sympy_extras.core import symbols, Eq, Rational, sqrt, cbrt, pi
from sympy_extras.functions import exp, log, sin, cos, tan, sinh, cosh, Abs
from sympy_extras.solvers.transcendental import solve_transcendental, polynomialize
from sympy_extras.solvers.inequalities import solve_inequality
assert 'sympy' not in sys.modules
x, = symbols('x,')
assert str(solve_transcendental(Eq(exp(2*x) - 3*exp(x) + 2, 0), x)) == '{0, log(2)}'
assert str(solve_transcendental(exp(x) + exp(-x) > 3, x)) == \
    'Union(Interval.open(-oo, log(3/2 - sqrt(5)/2)), Interval.open(log(sqrt(5)/2 + 3/2), oo))'
assert str(solve_transcendental(Eq(sin(x) + cos(x), 1), x)) == \
    'Union(ImageSet(Lambda(_k, 2*_k*pi), Integers), ImageSet(Lambda(_k, 2*_k*pi + pi/2), Integers))'
assert str(solve_transcendental(Eq(sin(2*x), Rational(1, 2)), x)) == \
    'Union(ImageSet(Lambda(_k, _k*pi + pi/12), Integers), ImageSet(Lambda(_k, _k*pi + 5*pi/12), Integers))'
assert str(solve_transcendental(Eq(log(x)**2, 3*log(x) - 2), x)) == '{E, exp(2)}'
assert str(solve_transcendental(Eq(sqrt(x + 1) + cbrt(x + 1), 2), x)) == '{0}'
assert str(solve_transcendental(Eq(Abs(x - 1) + Abs(x + 1), 3), x)) == '{-3/2, 3/2}'
assert str(solve_transcendental(Eq(sinh(x), 2), x)) == '{log(2 + sqrt(5))}'
assert str(solve_transcendental(Eq(cosh(2*x), 3), x)) == '{-log(3 - 2*sqrt(2))/2, -log(2*sqrt(2) + 3)/2}'
assert str(solve_transcendental(Eq(2**x, 8), x)) == '{3}' and str(solve_transcendental(Eq(exp(x), pi), x)) == '{log(pi)}'
assert str(solve_transcendental(Abs(exp(x) - 3) < 1, x)) == 'Interval.open(log(2), log(4))'
assert str(solve_transcendental(Eq(tan(x), 1), x)).startswith('Union(ImageSet(Lambda(_k, 2*_k*pi - 3*pi/4)')
assert str(polynomialize(Eq(exp(x/2) + exp(x), 6), x)) == 'Reduction(Eq(_t**2 + _t - 6, 0), _t = exp(x/2))'
assert str(solve_inequality(x**2 - 4 < 0, x)) == '(-2 < x) & (x < 2)'
assert str(solve_inequality(x < x*(2/x - 1), x)) == '(x < 1) & Ne(x, 0)'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_LINEAR_ODE = r'''
import sys
from sympy_extras.core import Function, Integer, Rational, Symbol
from sympy_extras.core.power import sqrt
from sympy_extras.functions import exp
from sympy_extras.solvers.linear_ode import (LinearOperator, UPoly, gcrd, hyperexponential_search,
    hyperexponential_solutions, lclm, log_derivative, polynomial_solutions, rational_solutions, reduce_order_linear)
from sympy_extras.solvers.factorization import factor_operator, left_factor, right_factor
from sympy_extras.solvers.irregular import Place, conjugate_sum, finite_exponential_parts, irregular_places
from sympy_extras.solvers.special import bessel_solutions, hypergeometric_solutions, whittaker_m
assert 'sympy' not in sys.modules
x = Symbol('x')
y = Function('y')(x)
zero, one = Integer(0), Integer(1)
D = LinearOperator([zero, one], x)
R = D - LinearOperator([1/x], x)
assert gcrd((D*D + LinearOperator([one], x))*R, (D - LinearOperator([x], x))*R) == R
assert lclm(D - LinearOperator([one], x), LinearOperator([-one, x], x)).order == 2
L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
assert polynomial_solutions(L) == [x**2 + 2*x + 2]
assert hyperexponential_solutions(L) == [x**2 + 2*x + 2, exp(x)]
assert rational_solutions(LinearOperator.from_equation(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)) == [x**-2, 1/x]
L = LinearOperator.from_equation(x**4*y.diff(x, 2) + 2*x**3*y.diff(x) - y, y)
assert finite_exponential_parts(L) == ([zero, 1/x, -1/x], True) and len(irregular_places(L)) == 1
assert hyperexponential_search(LinearOperator([-one, zero, x**2], x)) == ([], False)
assert reduce_order_linear(LinearOperator([Integer(2), Integer(-3), one], x), exp(x)) == LinearOperator([-one, one], x)
P = Place(UPoly.of(x**2 - 2, x), x)
assert conjugate_sum(P, P.alpha, 2) == 8*x/(x**2 - 2)**2
assert log_derivative(sqrt(x)*exp(x), x) == (2*x + 1)/(2*x)
F = factor_operator(LinearOperator([zero, -one, zero, one], x))
assert len(F.factors) == 3 and F.is_complete
assert left_factor(R*(D*D + LinearOperator([one], x)), 1) == R
assert right_factor((D*D + LinearOperator([one], x))*R, 1) == R
assert bessel_solutions(-x, x) is not None and hypergeometric_solutions(x, x) is None
assert str(whittaker_m(zero, Rational(1, 2), x)) == 'x*exp(-x/2)*hyper((1,), (2,), x)'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_lie_and_pde_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.solvers.lie` and
    :mod:`sympy_extras.solvers.pde` does not load SymPy, nor do the
    symmetries with parameters, a basis of functions and a quadratic
    leading derivative, their checks, the Janet basis of the determining
    system, the similarity reductions which need no ODE solution and the
    check of a solution (SymPy's ``dsolve``, and ``solve`` for equations
    which are not linear in the unknown, go through the faces at run
    time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_LIE_PDE], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_transcendental_and_inequalities_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.solvers.transcendental` and
    :mod:`sympy_extras.solvers.inequalities` does not load SymPy, nor do
    the equations and inequalities in exponentials, logarithms, sines and
    cosines, roots and absolute values whose reduced polynomial formulas
    are rational or linear, and the relationals rational in the variable
    (SymPy's ``solveset`` of the other polynomial relations, the Lambert
    W candidates of its ``solve``, the reliable numerical values and
    SymPy's ``_solve_inequality`` of the other relationals go through it
    at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_TRANSCENDENTAL], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_linear_ode_cluster_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.solvers.linear_ode`,
    :mod:`~sympy_extras.solvers.factorization`,
    :mod:`~sympy_extras.solvers.irregular` and
    :mod:`~sympy_extras.solvers.special` does not load SymPy, nor do the
    operator arithmetic, the polynomial, rational and hyperexponential
    solutions, the exponential parts at irregular points, Beke's
    factorisation of an operator whose factors are of order one and the
    special function forms (SymPy's ``integrate``, ``dsolve`` and
    ``solve``, which Beke's method asks for the Grassmann–Plücker
    relations of several solutions, go through SymPy at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_LINEAR_ODE], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_LINEAR_SYSTEMS = r'''
import sys
from sympy_extras.core import Integer, Symbol
from sympy_extras.functions import exp
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys.polytools import cancel
from sympy_extras.solvers.linear_ode import LinearOperator
from sympy_extras.solvers.linear_systems import (cyclic_reduction, cyclic_vector, dsolve_linear_system, dsolve_system,
    hyperexponential_system_solutions, rational_system_solutions, system_to_scalar)
assert 'sympy' not in sys.modules
x = Symbol('x')
E = ExprMatrix.from_rows
c, L = cyclic_vector(E([[1, 0], [0, 2]]), x)
assert str((c, L)) == '(ExprMatrix([[2, -3]]), LinearOperator([2, -3, 1], x))', (c, L)
L = LinearOperator([Integer(2), -2*x, x**2], x)
assert system_to_scalar(E([[1/x, 1], [0, 1/x]]), x) == (L, E([[1, 0], [1/x, 1]]))
R = cyclic_reduction(E([[0, 1], [2/x**2, 0]]), x)
assert R.to_system(x**2) == ExprMatrix.column([x**2, 2*x])
assert rational_system_solutions(E([[1/x, 1], [0, 1/x]]), x) == [ExprMatrix.column([x, 0]),
                                                             ExprMatrix.column([x**2, x])]
assert len(rational_system_solutions(E([[-1/x, 0], [1/x, -3/x]]), x)) == 2
assert rational_system_solutions(E([[1/x**2, 0], [0, -1/x**2]]), x) == []
assert hyperexponential_system_solutions(E([[1, 1/x], [0, 0]]), x) == [ExprMatrix.column([exp(x), 0])]
S = dsolve_system(E([[0, 1], [2/x**2, 0]]), x)
assert S.complete is True and S.fundamental == E([[1/x, x**2], [-1/x**2, 2*x]])
S = dsolve_system(E([[1, 1], [4, 1]]), x)
assert S.complete is True and S.fundamental == E([[exp(3*x), exp(-x)], [2*exp(3*x), -2*exp(-x)]]), S
assert len(dsolve_linear_system(E([[0, 1], [-1, 0]]), x)) == 2
A = E([[0, 1], [2/x**2, 0]])
Yp = ExprMatrix.column([1/(x + 1), x**3])
b = (Yp.diff(x) - A*Yp).applyfunc(cancel)
S = dsolve_system(A, x, b)
assert S.complete is True and S.particular is not None
assert all(e.is_rational_function(x) for e in S.particular.entries)
# a double pole with det N = 0 (A singular): the constant solution (1, -1)
assert len(rational_system_solutions(E([[1/x**2, 1/x**2], [1/x**2, 1/x**2]]), x)) == 1
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_DARBOUX = r'''
import sys
from sympy_extras.core import Function, I, Integer, Symbol, sqrt
from sympy_extras.solvers.darboux import (darboux_first_integral, darboux_integrating_factor, darboux_polynomials,
    exponential_factors, prelle_singer, vector_field)
assert 'sympy' not in sys.modules
x, Y, a, b = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b')
y = Function('y')(x)
assert str(darboux_polynomials(Y*(x - 1), x*(1 - Y), x, Y)) == '[DarbouxPolynomial(y, x - 1), DarbouxPolynomial(x, 1 - y)]'
assert str(darboux_polynomials(x + Y, x - Y, x, Y, degree=1, extension=I)) == \
    '[DarbouxPolynomial(x - I*y, 1 - I), DarbouxPolynomial(x + I*y, 1 + I)]'
assert str(darboux_polynomials(sqrt(2)*Y, x, x, Y, degree=1)) == '[DarbouxPolynomial(y, sqrt(2)), DarbouxPolynomial(x, 1)]'
assert str(darboux_polynomials(b - a*Y**2, 1, x, Y)) == '[DarbouxPolynomial(y**2 - b/a, -2*a*y)]'
assert str(exponential_factors(x*Y + 1, 1, x, Y, degree=1)) == '[ExponentialFactor(x, 1, 1), ExponentialFactor(x**2/2, 1, x)]'
assert str(darboux_integrating_factor(x*Y + 1, 1, x, Y, degree=1)) == 'exp(-x**2/2)'
assert str(darboux_first_integral(Y*(x - 1), x*(1 - Y), x, Y)) == '-x - y + log(x) + log(y)'
assert str(darboux_first_integral(x*Y**2 - Y, x**2*Y - x, x, Y, degree=2)) == 'x/y'
assert str(darboux_first_integral(x + Y, x - Y, x, Y, degree=1)) == '(x + I*y)/(x - I*y)**I'
assert darboux_first_integral(Y**2 + x, 1, x, Y, degree=2) is None
assert str(vector_field((x - y)*y.diff(x) - x - y, y)) == '(x + y, x - y, x, y)'
assert str(prelle_singer(x*(1 - y)*y.diff(x) - y*(x - 1), y)) == 'Eq(-x - y(x) + log(x) + log(y(x)), C1)'
assert str(prelle_singer(y.diff(x) - (x*y**2 - y)/(x**2*y - x), y)) == 'Eq(y(x), x/C1)'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_FIRST_ORDER = r'''
import sys
from sympy_extras.core import Eq, Function, Integer, Rational, symbols
from sympy_extras.solvers.first_order import (_derivative_polynomial, _explicit_rhs, abel_ode, chini_ode,
    dsolve_first_order, lagrange_ode, riccati_ode)
from sympy_extras.solvers.abel import (_defines_solution, _transformed, abel_coefficients, abel_invariants,
    constant_invariant)
assert 'sympy' not in sys.modules
x, t, C1 = symbols('x t C1')
y = Function('y')(x)
F, _, u, p = _derivative_polynomial(Eq(y.diff(x), x*y**2), y)
assert _explicit_rhs(F, p) == x*u**2
assert abel_coefficients(y.diff(x) - x*y**3 - y**2 + 1, y) == ((x, Integer(1), Integer(0), Integer(-1)), x)
assert str(abel_invariants((-1, -2*x, 0, 0), x).I1) == '11664*x**6*(8*x**3 - 15)**3/(8*x**3 - 9)**5'
assert abel_invariants((1, 0, 0, x**Rational(-3, 2)), x).I1 == Rational(-27, 8)
assert constant_invariant((-1, -2*x, 0, 0), x) is False
airy = (Integer(-1), -2*t, Integer(0), Integer(0))
assert str(_transformed(airy, t, x**2, x, Integer(1), x)) == \
    '(-2*x**3, -4*x**4 - 6*x**2, (-8*x**4 - 6*x**2 - 1)/x, -4*x**2 - 2)'
# a relation without x implies y' = 0 (sympy-extras#38); 1/y**2 + 2 x = C1 solves y' = y**3
assert not _defines_solution(y - C1, (x, Integer(1), Integer(0), Integer(0)), x, y, C1)
assert _defines_solution(1/y**2 + 2*x - C1, (Integer(1), Integer(0), Integer(0), Integer(0)), x, y, C1)
assert all(found is None for found in (chini_ode(y.diff(x) - y**3 - x, y), abel_ode(y.diff(x) - y**2, y),
                                       lagrange_ode(y.diff(x) - y**2, y), riccati_ode(y.diff(x) - y**3, y),
                                       lagrange_ode(Eq(y, x*y.diff(x) + y.diff(x)**2), y),
                                       chini_ode(y.diff(x) - y**2*x - y**3, y)))
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_linear_systems_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.solvers.linear_systems` does not load
    SymPy, nor do the cyclic vectors and their scalar equations, the
    rational solutions (Barkatou's bounds), the hyperexponential ones, the
    fundamental matrices which the scalar solvers find without quadratures
    and a rational particular solution (SymPy's ``integrate``, its rational
    integrator with ``RootSum`` and ``linodesolve`` go through
    :mod:`~sympy_extras.solvers.linear_systems_sympy` at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_LINEAR_SYSTEMS], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_darboux_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.solvers.darboux` does not load SymPy,
    nor do the Darboux polynomials over the rationals, number fields and
    fields of parameters, the exponential factors, the integrating
    factors, the first integrals from a vanishing combination of the
    cofactors (rational or logarithmic, the search repeated over the
    field of the conjugate lines), the parsing of an equation and an
    implicit or explicit solution from such a first integral (SymPy's
    ``integrate``, for the quadratures of an integrating factor, goes
    through SymPy at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_DARBOUX], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_first_order_and_abel_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.solvers.first_order` and
    :mod:`sympy_extras.solvers.abel` does not load SymPy, nor do the
    reading of an equation, the coefficients, the invariants and the
    transformations of Abel equations, the decision that the invariant is
    not constant, the numerical check of an implicit relation and the
    equations outside the classes recognised (SymPy's ``integrate``,
    ``dsolve`` and ``solve``, and the Prelle–Singer procedure of
    :mod:`~sympy_extras.solvers.darboux`, go through the faces at run
    time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_FIRST_ORDER], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_mellin_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.integrals.mellin` does not load SymPy,
    nor do the table of transforms, the matching of the integrands, the
    scale conditions decided by the native ``ask`` and the operations on
    the gamma quotients."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_MELLIN], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_ODE = r'''
import sys
from sympy_extras.core import Eq, Function, Symbol, symbols
from sympy_extras.functions import exp
from sympy_extras.solvers.lie import JetSpace, Symmetry
from sympy_extras.solvers.ode import canonical_coordinates, check_ode_solution, ode_symmetries
from sympy_extras.solvers.second_order import (commuting_pair, integrating_factor_p, integrating_factor_xy,
    is_linearizable, linearize, second_order_rhs)
assert 'sympy' not in sys.modules
x, yy, p, C1 = symbols('x y p C1')
y = Function('y')(x)
assert len(ode_symmetries(y.diff(x, 2), y)) == 8
assert [X.generator() for X in ode_symmetries(y.diff(x, 2) + 2*y.diff(x)/x + y**5, y)] == ['x*d/dx - y/2*d/dy']
jets = JetSpace([y])
u = jets.u[0]
assert canonical_coordinates(Symmetry(jets, [0], [0])) is None
assert check_ode_solution(y.diff(x) - y, Eq(y, C1*exp(x)), y)
assert not check_ode_solution(y.diff(x) - y, Eq(y, C1*exp(2*x)), y)
assert check_ode_solution(y.diff(x, 2) - y, C1*exp(-x), y)
assert str(second_order_rhs(x*y.diff(x, 2) + y.diff(x) - y**2, y)[0]) == '(-_p + _y**2)/x'
assert is_linearizable(-3*yy*p - yy**3, x, yy, p) and not is_linearizable(yy**2, x, yy, p)
assert linearize(yy**2, x, yy, p) is None
assert integrating_factor_xy(p**3, x, yy, p) is None and integrating_factor_p(p**3*(yy**2 + x*p), x, yy, p) is None
basis = [Symmetry(jets, [1], [1]), Symmetry(jets, [1], [-1]), Symmetry(jets, [x], [u])]
assert commuting_pair(basis, x, u) is not None
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_SLATER_MARICHEV = r'''
import sys
from sympy_extras.core import Integer, S, symbols, sqrt
from sympy_extras.functions import besselj, exp, exp_polar, log, meijerg, sin
from sympy_extras.integrals.mellin import GammaQuotient
from sympy_extras.integrals.slater import expand_meijerg, line_conditions, mellin_barnes, slater_expansion
from sympy_extras.integrals.marichev import checked_tidy, mellin_integrate
x, k = symbols('x k')
a, b, z = symbols('a b z', positive=True)
g = mellin_barnes(GammaQuotient(1, [(3, -1)], [(0, S.One), (1, -S.One)], [], 0, 1))
assert str(g) == 'MeijerG(1, meijerg(((1,), ()), ((1,), ()), 1/3))'
assert str(expand_meijerg(g)) == 'ConditionalValue(1/4)' and line_conditions(g, Integer(0), Integer(1)) is S.true
assert str(slater_expansion(meijerg([1 - a], [], [0], [], z))) == 'gamma(a)/(z + 1)**a'
assert str(mellin_integrate(exp(-a*x)*sin(b*x), x)) == 'ConditionalValue(b/(a**2 + b**2))'
assert str(mellin_integrate(x**k/(x + 3), x)) == 'ConditionalValue(-3**k*pi/sin(pi*k), (k > -1) & (k < 0))'
assert str(mellin_integrate(exp(-x**2)*log(x), x)) == 'ConditionalValue(sqrt(pi)*(-log(4) - EulerGamma)/4)'
assert str(mellin_integrate(x**(b - 1)*(1 - x)**(a - 1), x, cutoff='lower')) == \
    'ConditionalValue(gamma(a)*gamma(b)/gamma(a + b))'
assert str(mellin_integrate(exp(-x)*besselj(Integer(0), x), x)) == 'ConditionalValue(sqrt(2)/2)'
assert str(checked_tidy(sqrt(a**2) + log(a*exp_polar(Integer(0))))) == 'a + log(a)'
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_BRACKETS_DIRICHLET = r'''
import sys
from sympy_extras.core import Integer, Rational, I, oo, symbols
from sympy_extras.core.symbol import Dummy
from sympy_extras.functions import atan, exp, factorial, rf, sin
from sympy_extras.functions.elementary.trigonometric import Cos, Sin
from sympy_extras.integrals.brackets import SeriesCoefficient, _factors, _order_at_infinity
from sympy_extras.integrals.dirichlet import dirichlet_integral, trigonometric_sum
x, a = symbols('x a')
assert str(dirichlet_integral(sin(x)**3/x**2, x, 0, oo)) == 'ConditionalValue(3*log(3)/4)'
assert str(dirichlet_integral(sin(x)*sin(3*x)*sin(5*x)*sin(7*x)/(105*x**4), x, -oo, oo)) == 'ConditionalValue(44*pi/315)'
assert dirichlet_integral(sin(x)**2/x, x, 0, oo) is None
assert str(dirichlet_integral(sin(a*x)/x, x, 0, oo, a < 0)) == 'ConditionalValue(-pi/2)'
assert trigonometric_sum(exp(2*I*x), x)[1] == {(Integer(2), Cos): Integer(1), (Integer(2), Sin): I}
j = Dummy('j', integer=True, nonnegative=True)
c = SeriesCoefficient((Integer(-1)/4)**j/(rf(Rational(3, 2), j)*factorial(j)), j, 2, Integer(1))
assert str(c.phi()) == 'sqrt(pi)/(2*2**(2*_j)*gamma(_j + 3/2))'
assert str(_factors(3*x**2*exp(-x)*sin(2*x), x)) == '(3, 2, [(exp(-_y), 1, 1), (sin(_y), 2, 1)])'
assert (_order_at_infinity(1/(1 + x)**2, x), _order_at_infinity(atan(x), x)) == (2, 0)
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_ode_and_second_order_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.solvers.ode` and
    :mod:`sympy_extras.solvers.second_order` does not load SymPy, nor do
    the symmetries of an equation, the check of an explicit solution (the
    port of SymPy's ``checkodesol``), the right-hand side of a second
    order equation, Lie's linearisation test, the integrating factors
    refused before a quadrature and a commuting pair of symmetries found
    in a centralizer (SymPy's ``dsolve``, ``solve`` and ``integrate`` go
    through the faces at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_ODE], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_brackets_and_dirichlet_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.integrals.brackets` and
    :mod:`sympy_extras.integrals.dirichlet` does not load SymPy, nor do the
    Dirichlet integrals (the product formulas, the native ``ask``,
    ``simplify`` and ``logcombine``) and the steps of the method of
    brackets after the formal power series (the coefficient function,
    the factors, the decay at infinity by the native series); SymPy's
    ``fps`` is reached through the face only when a series is needed."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_BRACKETS_DIRICHLET], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_CONVERGENCE = r'''
import sys
from sympy_extras.core import symbols
from sympy_extras.functions import sin
from sympy_extras.concrete.convergence import is_convergent, product_convergence, sum_convergence
assert 'sympy' not in sys.modules
n = symbols('n')
assert str(sum_convergence(0, n)) == 'True' and str(sum_convergence(1, n)) == 'False'
assert str(sum_convergence(1/n**2, n)) == 'True' and str(sum_convergence(1/n, n)) == 'False'
assert str(sum_convergence(((-1)**n - 1)/n**3, n)) == 'True'
assert str(sum_convergence((sin(n) + 3)/(n**2 + 1), n)) == 'True'
assert str(sum_convergence(2**(-n), n)) == 'True'
assert str(product_convergence(1, n)) == 'True' and str(product_convergence(1 + 1/n**2, n)) == 'True'
assert is_convergent(1/(n**2 + 1), n) is True
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


_NATIVE_HELPERS = r'''
import sys
import sympy_extras._typing, sympy_extras._timeout, sympy_extras._special_values
assert 'sympy' not in sys.modules
from sympy_extras._timeout import attempt, time_limit, remaining_time
from sympy_extras._special_values import (special_points, with_special_values, condition_point, point_assumptions,
    undefined_at, equality_points, defined_problem, point_condition)
from sympy_extras.core import Eq, Function, Integer, Rational, symbols, sqrt
from sympy_extras.functions import exp, log, sin, zeta, DiracDelta
from sympy_extras.solvers._special_values import general_solution_cases
a, b, k, y, x, z, C1, C2 = symbols('a b k y x z C1 C2')
assert special_points((y**2 - 1)/log(y), [y]) == [{y: 0}, {y: 1}]
assert special_points(1/(a**2 - b**2), [a, b]) == [{a: -b}, {a: b}]
assert special_points(1/(a**2 - 2), [a]) == [{a: -sqrt(2)}, {a: sqrt(2)}]
assert special_points((exp(k) - 1)/k + 1/(exp(a) - 1) + zeta(b), [a, b, k]) == [{a: 0}, {b: 1}, {k: 0}]
assert str(with_special_values((y**2 - 1)/log(y), [y], None, lambda p, at: Integer(2))) == \
    'Piecewise((2, Eq(y, 1)), ((y**2 - 1)/log(y), True))'
assert str(with_special_values(exp(k)/k, [k], None, lambda p, at: Integer(1), sympy_style=True)) == \
    'Piecewise((exp(k)/k, Ne(k, 0)), (1, True))'
assert with_special_values(1/a, [a], a > 0, lambda p, at: Integer(1)) == 1/a
assert condition_point(Eq(a, b) & Eq(b, 0)) == {a: 0, b: 0} and point_assumptions({a: Integer(0)}, a > 0) is None
assert undefined_at(sin(a)/a, {a: Integer(0)}) and not defined_problem(DiracDelta(a*z).subs(a, 0), [z])
assert equality_points(with_special_values(1/(a - 1), [a], None, lambda p, at: Integer(0)), [a]) == [{a: 1}]
f = Function('f')(x)
equation = f.diff(x, 2) - (a + b)*f.diff(x) + a*b*f
assert str(general_solution_cases(C1*exp(a*x) + C2*exp(b*x), equation, f, lambda e: (C1 + C2*x)*exp(b*x))) == \
    'Piecewise(((C1 + C2*x)*exp(b*x), Eq(a, b)), (C1*exp(a*x) + C2*exp(b*x), True))'
def slow() -> int:
    while True:
        pass
assert attempt(slow, 0.1, native=True) is None and remaining_time() is None
with time_limit(10, native=True):
    assert 0 < (remaining_time() or 0) <= 10
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_convergence_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.concrete.convergence` does not load
    SymPy, nor do the comparison with a rational majorant, the term test
    and the ratio and Raabe tests on series without parameters, and the
    products reduced to them (SymPy's ``integrate`` of the integral test,
    its ``Sum.is_convergent`` and the steps of the native assumptions which
    stay on SymPy go through the faces at run time)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_CONVERGENCE], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_slater_and_marichev_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.integrals.slater` and
    :mod:`sympy_extras.integrals.marichev` does not load SymPy, nor do the
    Mellin–Barnes conversion, Slater's theorem with the native
    ``hyperexpand``, the Marichev–Adamchik integrals of one and two kernels
    and the simplification of their values."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_SLATER_MARICHEV], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_native_helpers_do_not_load_sympy() -> None:
    """Importing the top-level helpers :mod:`sympy_extras._typing`,
    :mod:`sympy_extras._timeout` and :mod:`sympy_extras._special_values`
    does not load SymPy, nor do the candidate points of the parameters
    (zeros of denominators, logarithms and ``zeta``, linear, quadratic and
    transcendental factors), the cases with their bookkeeping, the cases
    of an ODE solution at a repeated characteristic root
    (:mod:`sympy_extras.solvers._special_values`, which no longer goes
    through SymPy for them) and native time limits."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_HELPERS], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_EXPONENTIAL_RECOGNIZE = r'''
import sys
from sympy_extras.core import Float, Rational, symbols, oo, pi, sqrt
from sympy_extras.functions import atan, exp, log
from sympy_extras.integrals.exponential import (composite_gaussian, exponential_antiderivative, power_exponential,
                                                rational_exponential, reciprocal_square_exponential,
                                                nested_power_exponential, exponential_rational)
from sympy_extras.integrals.recognize import quadrature, recognize_constant, recognize_integral
assert 'sympy' not in sys.modules
x, c = symbols('x c')
v = symbols('v', positive=True)
a = symbols('a', negative=True)
assert str(exponential_antiderivative(x**(v - 1)*exp(a*x**2), x)) == '-uppergamma(v/2, -a*x**2)/(2*(-a)**(v/2))'
assert str(exponential_antiderivative(x*exp(x**2 + 2*x), x)) == 'exp(-1)*exp((x + 1)**2)/2 - sqrt(pi)*exp(-1)*erfi(x + 1)/2'
assert str(power_exponential(x**3*exp(x**2), x)) == 'x**2*exp(x**2)/2 - exp(x**2)/2'
assert exponential_antiderivative(exp(x**3), x, [x > 0]) is not None
assert str(rational_exponential(exp(c*x)/(x**2 - 1), x)) == 'exp(c)*Ei(c*(x - 1))/2 - exp(-c)*Ei(c*(x + 1))/2'
assert str(composite_gaussian((1 - 1/x**2)*exp(-(x + 1/x)**2), x)) == 'sqrt(pi)*erf((x**2 + 1)/x)/2'
assert str(reciprocal_square_exponential(exp(-x**2 - 1/x**2), x)) == 'sqrt(pi)*(exp(-2)*erf(x - 1/x) + exp(2)*erf(x + 1/x))/4'
assert str(nested_power_exponential(exp(c*sqrt(x**2)), x)) == 'x*exp(c*sqrt(x**2))/(c*sqrt(x**2))'
assert str(recognize_constant(Float('1.6449340668482264364724151666460251892', 38))) == 'pi**2/6'
assert str(recognize_integral(1/(x**3 + 1), (x, 0, 1))) == 'log(2)/3 + sqrt(3)*pi/9'
assert str(exponential_rational(1/(exp(x) + 1), x)) == 'x - log(exp(x) + 1)'
assert str(exponential_rational(exp(x)/sqrt(exp(x) + 1), x)) == '2*sqrt(exp(x) + 1)'
assert str(recognize_integral(atan(x)/x, (x, 0, 1))) == 'Catalan'
assert quadrature(exp(-x**2), x, -oo, oo) is not None
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_exponential_and_recognize_do_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.integrals.exponential` and
    :mod:`sympy_extras.integrals.recognize` does not load SymPy, nor do the
    antiderivatives of powers times exponentials of powers, of shifted
    quadratics, of rational functions times an exponential (the native
    ``roots``), of composite and reciprocal-square Gaussians and of nested
    powers (the native ``hyperexpand``, ``simplify`` and numerical checks),
    the quadrature and the PSLQ recognition, nor the rational functions of
    ``exp(c*x)`` (the native ``ratint``)."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_EXPONENTIAL_RECOGNIZE], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


_NATIVE_RATINT = r'''
import sys
from sympy_extras.core import symbols, Lambda, I
from sympy_extras.functions import log
from sympy_extras.integrals.risch.rationaltools import ratint, ratint_ratpart, ratint_logpart, log_to_real
from sympy_extras.polys.rootoftools import RootSum, root_sum
from sympy_extras.solvers.linear_systems import rational_integral
x, a, b, t = symbols('x a b t')
p = symbols('p', positive=True)
assert str(ratint(36/(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2), x)) == '(12*x + 6)/(x**2 - 1) + 4*log(x - 2) - 4*log(x + 1)'
assert str(ratint(1/(x**2 + x + 1), x)) == '2*sqrt(3)*atan(2*sqrt(3)*x/3 + sqrt(3)/3)/3'
assert str(ratint(1/(x*(a + b*x)**3), x)) == '(3*a + 2*b*x)/(2*a**4 + 4*a**3*b*x + 2*a**2*b**2*x**2) + (log(x) - log(a/b + x))/a**3'
assert str(ratint(1/(x**2 + a**2), x)) == '(-I*log(-I*a + x)/2 + I*log(I*a + x)/2)/a'
assert str(ratint(1/(x**2 + p**2), x)) == 'atan(x/p)/p'
F = ratint(1/(x**3 - x - 1), x, real=False)
assert isinstance(F, RootSum)
assert str(F.diff(x)) == '1/(x**3 - x - 1)', F.diff(x)
assert ratint(1/(x**3 - x - 1), x).has(RootSum) is False
assert str(ratint_ratpart(36, x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2, x)) == '((12*x + 6)/(x**2 - 1), 12/(x**2 - x - 2))'
assert str(ratint_logpart(1, x**2 + x + 1, x, t)) == '[(3*t/2 + x + 1/2, 3*t**2 + 1)]'
# the value of SymPy's RootSum(...).subs(x, 2).evalf(10)
assert str(root_sum(t**3 - t - 1, Lambda(t, t*log(x + t)), t).subs(x, 2).evalf(10)) == '0.6508608717'
assert str(rational_integral(1/(x**3 - x - 1) + 1/x**2, x)).endswith('- 1/x')
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_native_ratint_does_not_load_sympy() -> None:
    """Importing :mod:`sympy_extras.integrals.risch.rationaltools`,
    :mod:`sympy_extras.polys.rootoftools` and running the rational
    integrator over ``QQ`` and with parameters (the domain ``EX``), its
    ``RootSum`` (the derivative by the trace, the numerical value) and the
    rational antiderivatives of :mod:`sympy_extras.solvers.linear_systems`
    does not load SymPy."""
    out = subprocess.run([sys.executable, '-c', _NATIVE_RATINT], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout
