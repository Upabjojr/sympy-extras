r"""The LaTeX printer on what the comparison with SymPy does not cover:
the settings which change the deliberate differences, the classes SymPy
does not have, the Python values, the errors."""
from __future__ import annotations

import subprocess
import sys
from fractions import Fraction

from sympy.testing.pytest import raises

from sympy_extras.assumptions.quantifiers import Exists, ForAll
from sympy_extras.concrete.qhyper import QPochhammer
from sympy_extras.core import (Add, And, Derivative, Dummy, Float, Function, I, Integer, Lambda, Mul, Not, Order,
    Pow, Q, Rational, Subs, Symbol, Tuple, Wild, oo, pi, sqrt)
from sympy_extras.core.function import WildFunction
from sympy_extras.core.limits import Limit
from sympy_extras.core.logic import ITE, Equivalent, Implies
from sympy_extras.functions.elementary.exponential import exp, log
from sympy_extras.functions.elementary.piecewise import ExprCondPair
from sympy_extras.functions.elementary.trigonometric import asin, cos, sin
from sympy_extras.integrals.integral import Integral
from sympy_extras.integrals.regions import IntegralByRanges
from sympy_extras.polys.cad.indexedroot import IndexedRoot
from sympy_extras.polys.complexrootof import CRootOf
from sympy_extras.polys.rootoftools import RootSum
from sympy_extras.printing import latex
from sympy_extras.printing.latex import LatexPrinter, LatexSettings
from sympy_extras.series.formal import fps
from sympy_extras.series.fourier import fourier_series
from sympy_extras.series.sequences import EmptySequence, RecursiveSeq, SeqAdd, SeqFormula, SeqMul, SeqPer
from sympy_extras.sets.conditionset import ConditionSet
from sympy_extras.sets.sets import FiniteSet, Interval, PointSet, SetOfSets, UniversalSet
from sympy_extras.solvers.isolation import TranscendentalRoot

x, y, z, n, k, a, b, t = (Symbol('x'), Symbol('y'), Symbol('z'), Symbol('n'), Symbol('k'), Symbol('a'), Symbol('b'),
                          Symbol('t'))
f = Function('f')


def test_short_fractions_on_one_line() -> None:
    # a denominator after a slash is a power or tighter: SymPy prints x / 2 y
    assert latex(x/(2*y), fold_short_frac=True) == r'x / \left(2 y\right)'
    assert latex(x/2, fold_short_frac=True) == 'x / 2' and latex(-x/2, fold_short_frac=True) == '- x / 2'
    assert latex((x + 1)/y, fold_short_frac=True) == r'\left(x + 1\right) / y'
    assert latex(x/y**2, fold_short_frac=True) == r'\frac{x}{y^{2}}'  # a long denominator stays built up
    assert latex(2**(x/2), fold_short_frac=True) == '2^{x / 2}'
    assert latex(Mul(x, Rational(1, 2)), fold_short_frac=True) == r'x \left(1 / 2\right)'
    assert latex(Pow(Rational(1, 2), x), fold_short_frac=True) == r'\left(1 / 2\right)^{x}'
    assert latex(1/sqrt(x), mode='inline') == r'$1 / \sqrt{x}$'
    assert latex(1/x, mode='inline') == '$1 / x$' and latex(1/(x + y), mode='inline') == r'$\frac{1}{x + y}$'
    assert latex(Integral(x, x, Integer(0), Integer(1)), mode='inline') == r'$\int_{0}^{1} x\, dx$'
    assert latex(Integral(x, x, Integer(0), Integer(1))) == r'\int\limits_{0}^{1} x\, dx'


def test_inverse_trigonometric_styles() -> None:
    assert latex(asin(x)) == r'\operatorname{asin}{\left(x \right)}'
    assert latex(asin(x), inv_trig_style='full') == r'\arcsin{\left(x \right)}'
    assert latex(asin(x), inv_trig_style='power') == r'\sin^{-1}{\left(x \right)}'
    assert latex(asin(x)**2, inv_trig_style='full') == r'\arcsin^{2}{\left(x \right)}'
    # SymPy: \sin^{-1}{\left(x \right)}^{2}
    assert latex(asin(x)**2, inv_trig_style='power') == r'\left(\sin^{-1}{\left(x \right)}\right)^{2}'


def test_orders_floats_and_names() -> None:
    e = x**2 + y*x + y**3
    assert latex(e) == r'x^{2} + x y + y^{3}'
    assert latex(e, order='grlex') == r'y^{3} + x^{2} + x y' and latex(e, order='lex') == latex(e)
    # the reversed orders and SymPy's old order (ValueError before)
    assert latex(e, order='rev-lex') == r'y^{3} + x y + x^{2}' and latex(e, order='rev-grlex') == r'x y + x^{2} + y^{3}'
    assert latex(e, order='old') == r'x y + x^{2} + y^{3}' and latex(1 - x, order='lex') == '- x + 1'
    raises(ValueError, lambda: latex(e, order='degree'))
    assert latex(Add(y, x), order='none') == 'y + x'
    assert latex(Float(123456.789), min=-2, max=3) == r'1.23456789 \cdot 10^{5}'
    assert latex(Float(0.5), full_prec=True) == '0.500000000000000' and latex(Float(0.5)) == '0.5'
    assert latex(2*I*x, imaginary_unit='rj') == r'2 \mathrm{j} x'
    assert latex(x + y, symbol_names={x: r'\xi', y: 'y^{*}'}) == r'\xi + y^{*}'
    assert latex(Pow(y, 2), symbol_names={y: 'y^{*}'}) == r'\left(y^{*}\right)^{2}'
    assert latex(Derivative(f(x), x), diff_operator='rd') == r'\frac{\mathrm{d}}{\mathrm{d} x} f{\left(x \right)}'
    assert latex(x*y*z*(x + 1)/(a*b), long_frac_ratio=1) == r'\frac{x y}{a b} z \left(x + 1\right)'
    assert latex(x**Rational(1, 3), itex=True) == r'\root{3}{x}'
    assert latex(x < y, itex=True, mode='equation') == r'$$x \lt y$$'
    assert latex(log(x), ln_notation=True) == r'\ln{\left(x \right)}'


def test_wrong_settings_are_refused() -> None:
    raises(ValueError, lambda: latex(x, mode='display'))
    raises(ValueError, lambda: latex(x, decimal_separator='dot'))
    raises(ValueError, lambda: latex(x, inv_trig_style='arc'))
    # ('old' was refused: it is one of SymPy's orders, and is supported now)
    raises(ValueError, lambda: latex(x, order='oldest'))
    assert latex(x + 1, order='old') == '1 + x'
    raises(ValueError, lambda: latex(x, mat_delim='{'))


def test_the_settings_are_not_modified_by_printing() -> None:
    settings = LatexSettings(fold_frac_powers=True, mode='inline')
    printer = LatexPrinter(settings)
    before = (printer.settings, printer.fold_short_frac, printer.separator, printer.number_separator)
    first = printer.doprint(x**Rational(2, 3)/sin(x)**2 + Limit(f(x), x, a + b))
    assert printer.doprint(x**Rational(2, 3)/sin(x)**2 + Limit(f(x), x, a + b)) == first
    assert (printer.settings, printer.fold_short_frac, printer.separator, printer.number_separator) == before
    assert printer.settings is settings


def test_names() -> None:
    assert latex(Dummy('t') + 1) == 't + 1' and latex(Wild('w') + 1) == 'w + 1'
    assert latex(WildFunction('F')) == r'\operatorname{F\_}'
    assert latex(Function('Gamma')(x)) == r'\Gamma{\left(x \right)}'
    assert latex(Function('f^2')(x)**2) == r'\left(f^{2}\right)^{2}{\left(x \right)}'
    assert latex(Symbol('xdot_1')) == r'\dot{x}_{1}' and latex(Symbol('Omega__a_b')) == r'\Omega^{a}_{b}'
    assert latex(Derivative(f(Symbol('x^i')), (Symbol('x^i'), 2))) == \
        r'\frac{d^{2}}{d \left(x^{i}\right)^{2}} f{\left(x^{i} \right)}'


def test_logic_and_quantifiers() -> None:
    p, q = x < y, y < z
    assert latex(Not(Implies(p, q))) == r'x < y \not\Rightarrow y < z'
    assert latex(Not(Equivalent(p, q))) == r'x < y \not\Leftrightarrow y < z'
    assert latex(And(Not(Implies(p, q)), p)) == r'\left(x < y \not\Rightarrow y < z\right) \wedge x < y'
    assert latex(Not(Not(p))) == r'\neg \neg \left(x < y\right)'
    assert latex(ITE(p, q, z < x)) == r'\operatorname{ITE}{\left(x < y, y < z, z < x \right)}'
    assert latex(Q.positive(x)) == r'\operatorname{Q}_{\text{positive}}\left(x\right)'
    assert latex(ForAll(x, Exists(y, x < y))) == r'\forall x\colon \exists y\colon x < y'
    assert latex(And(ForAll(x, x < y), q)) == r'\left(\forall x\colon x < y\right) \wedge y < z'


def test_calculus_forms() -> None:
    assert latex(Subs(f(x, y), (x, y), (1, 2))) == r'\left. f{\left(x,y \right)} \right|_{\substack{ x=1\\ y=2 }}'
    assert latex(Order(x*y, x, y)) == r'O\left(x y; \left( x, \  y\right)\rightarrow \left( 0, \  0\right)\right)'
    assert latex(Derivative(f(x), (x, 3))) == r'\frac{d^{3}}{d x^{3}} f{\left(x \right)}'
    assert latex(Limit(1/x, x, oo)) == r'\lim_{x \to \infty} \frac{1}{x}'
    assert latex(Limit(x + 1, x, 0)) == r'\lim_{x \to 0^+} \left(x + 1\right)'
    assert latex(Lambda(x, x**2)) == r'\left( x \mapsto x^{2} \right)'
    region = IntegralByRanges(x*y, (x > 0) & (x < y) & (y < 1))
    assert latex(region).startswith(r'\int\limits_{') and latex(region).endswith(r'x y\, dx\, dy')


def test_powers() -> None:
    A, B = Symbol('A', commutative=False), Symbol('B', commutative=False)
    assert latex(Mul(A, Pow(B, -1))) == 'A B^{-1}'  # no fraction of noncommutative factors
    assert latex(Pow(-1, x)) == r'\left(-1\right)^{x}' and latex(Pow(x, I)) == 'x^{i}'
    assert latex(Pow(2, Pow(3, 4))) == '2^{3^{4}}' and latex(Pow(Pow(2, 3), 4)) == r'\left(2^{3}\right)^{4}'
    assert latex(Pow(Pow(sin(x), 3), -2)) == r'\frac{1}{\left(\sin^{3}{\left(x \right)}\right)^{2}}'
    assert latex(Pow(Pow(sin(x), 3), 2)) == r'\left(\sin^{3}{\left(x \right)}\right)^{2}'
    assert latex(Pow(exp(x), 2)) == r'\left(e^{x}\right)^{2}' and latex(exp(1)) == 'e'
    assert latex(sqrt(sqrt(x))) == r'\sqrt[4]{x}' and latex(Pow(sqrt(x), 2)) == r'\left(\sqrt{x}\right)^{2}'
    assert latex(x**Rational(2, 3), root_notation=False) == r'x^{\frac{2}{3}}'
    assert latex(x**Rational(1, 3), root_notation=False) == r'x^{\frac{1}{3}}'


def test_sequences_and_series() -> None:
    assert latex(SeqFormula(n**2, (n, 0, oo))) == r'\left[0, 1, 4, 9, \ldots\right]'
    assert latex(SeqFormula(n**2, (n, 0, 3))) == r'\left[0, 1, 4, 9\right]'
    assert latex(SeqFormula(n**2, (n, a, b))) == r'\left\{n^{2}\right\}_{n=a}^{b}'
    assert latex(SeqFormula(n**2, (n, -oo, 0))) == r'\left[\ldots, 9, 4, 1, 0\right]'
    assert latex(SeqPer((1, 2, 3), (n, 0, oo))) == r'\left[1, 2, 3, 1, \ldots\right]'
    squares, alternating = SeqFormula(n**2, (n, 0, oo)), SeqPer((1, 2), (n, 0, oo))
    assert latex(SeqAdd(squares, alternating)) == r'\left[1, 3, 5, 11, \ldots\right]'
    assert latex(SeqMul(squares, alternating)) == r'\left[0, 2, 4, 18, \ldots\right]'
    assert latex(EmptySequence()) == r'\left[\right]'
    u = Function('u')
    fibonacci = RecursiveSeq(u(n - 1) + u(n - 2), u(n), n, [0, 1])
    assert latex(fibonacci).startswith(r'\left\{u{\left(n \right)}\; \middle|\; ')
    assert latex(fourier_series(x, (x, -pi, pi))) == \
        r'2 \sin{\left(x \right)} - \sin{\left(2 x \right)} + \frac{2 \sin{\left(3 x \right)}}{3} + \ldots'
    assert latex(fps(log(1 + x))).startswith(r'\sum_{k=1}^{\infty} ')


def test_roots_and_other_objects() -> None:
    assert latex(CRootOf(x**5 + x + 1, 0)) == r'\operatorname{CRootOf} {\left(x^{5} + x + 1, 0\right)}'
    assert latex(RootSum(t**3 + t + 1, Lambda(t, t*log(x - t)), t)).startswith(r'\operatorname{RootSum} {\left(')
    assert latex(IndexedRoot(x**3 - y, x, 1)) == r'\operatorname{Root}_{1} {\left(x^{3} - y, x\right)}'
    assert latex(TranscendentalRoot(cos(x) - x, x, 0, 1)) == \
        r'\operatorname{Root} {\left(- x + \cos{\left(x \right)}, x \in \left(0, 1\right)\right)}'
    assert latex(QPochhammer(a, x, n)) == r'\left(a; x\right)_{n}'
    assert latex(Pow(QPochhammer(a, x, n), 2)) == r'\left(\left(a; x\right)_{n}\right)^{2}'
    assert latex(ExprCondPair(x, x < 1)) == r'\left( x, \  x < 1\right)'
    assert latex(PointSet(Tuple(1, 2), Tuple(3, 4))) == r'\left\{\left( 1, \  2\right), \left( 3, \  4\right)\right\}'
    assert latex(SetOfSets(FiniteSet(1), FiniteSet(1, 2))) == r'\left\{\left\{1\right\}, \left\{1, 2\right\}\right\}'
    assert latex(ConditionSet(x, x**2 > 1, UniversalSet)) == r'\left\{x\; \middle|\; x^{2} > 1 \right\}'
    assert latex(Interval(-oo, oo)) == r'\left(-\infty, \infty\right)' and latex(Interval(1, 1)) == r'\left\{1\right\}'


def test_python_values() -> None:
    assert latex({x: Integer(1)}) == r'\left\{ x : 1\right\}'
    assert latex([x, {y}, (1, 2.5), None, 'a_b']) == \
        r'\left[ x, \  \left\{y\right\}, \  \left( 1, \  2.5\right), \  \text{None}, \  \mathtt{\text{a\_b}}\right]'
    assert latex(frozenset([x])) == r'\left\{x\right\}' and latex(Fraction(-1, 2)) == r'- \frac{1}{2}'
    assert latex(True) == r'\text{True}' and latex(-3) == '-3' and latex([]) == r'\left[ \right]'


_INDEPENDENT = r'''
import sys
from sympy_extras.core import Symbol, Rational
from sympy_extras.functions.elementary.trigonometric import sin
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.printing import latex, init_printing
x = Symbol('x')
init_printing(display_hook=False)
shown = latex(sin(x)**2/2 + x**Rational(1, 3)), latex(ExprMatrix.from_rows([[x]])), (x/2)._repr_latex_()
assert 'sympy' not in sys.modules, sorted(m for m in sys.modules if m.startswith('sympy.'))[:5]
print(shown[2])
'''


def test_the_printers_do_not_load_sympy() -> None:
    out = subprocess.run([sys.executable, '-c', _INDEPENDENT], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip() == r'$\displaystyle \frac{x}{2}$'
