"""The LaTeX printer and the pretty printer stand on one framework: one
scale of bindings, one list of positions, one ``needs_brackets``, the
same conventions of names and the same order of the terms; and the
interactive session shows the results with both."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from fractions import Fraction

import sympy

from sympy_extras.core import Add, Float, Integer, Mul, Not, Pow, Rational, Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import ITE, And, Equivalent, Implies
from sympy_extras.functions.elementary.exponential import exp
from sympy_extras.functions.elementary.trigonometric import cos, sin
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.matrices.domainmatrix import DomainMatrix
from sympy_extras.polys.domains import QQ, ZZ
from sympy_extras.polys.polytools import Poly
from sympy_extras.printing import _latex_names, conventions, framework, init_printing, latex, pprint, pretty
from sympy_extras.printing.framework import Binding, Context, Dispatcher, Position, Result, TreePrinter, needs_brackets
from sympy_extras.printing.interactive import (native_pretty, pretty_printer, repr_pretty, reset_printing,
    set_pretty_printer, text_form)
from sympy_extras.printing.latex import RULES as LATEX_RULES, LatexInput, LatexPrinter, LatexSettings
from sympy_extras.printing.pretty import (Binding as PrettyBinding, Context as PrettyContext, Position as PrettyPosition,
    engine, layout, needs_brackets as pretty_needs_brackets, symbology)
from sympy_extras.printing.pretty.engine import PrettyPrinter, PrettySettings
from sympy_extras.printing.pretty.layout import TOP, Picture
from sympy_extras.printing.pretty.pretty import RULES as PRETTY_RULES, pretty_use_unicode

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')


class _Sink:
    def __init__(self) -> None:
        self.written: list[str] = []

    def text(self, s: str) -> None:
        self.written.append(s)


def _shown(e: Basic) -> str:
    sink = _Sink()
    e._repr_pretty_(sink, False)
    return ''.join(sink.written)


# ----------------------------------------------------------------------
# one framework

def test_the_pretty_printer_has_no_framework_of_its_own() -> None:
    assert PrettyBinding is Binding and PrettyPosition is Position
    assert PrettyContext is Context and pretty_needs_brackets is needs_brackets
    for module in (layout, engine):
        for name in ('Binding', 'Position', 'needs_brackets'):
            assert name not in vars(module) or vars(module)[name] is vars(framework)[name]
        assert 'Applied' not in vars(module)
    assert issubclass(PrettyPrinter, TreePrinter) and issubclass(LatexPrinter, TreePrinter)
    assert isinstance(PRETTY_RULES, Dispatcher) and isinstance(LATEX_RULES, Dispatcher)
    # the result of a rule of either printer is the Result of the framework
    drawn = PrettyPrinter(PrettySettings(), PRETTY_RULES).rule(-x/y, TOP)
    written = LatexPrinter(LatexSettings()).rule(-x/y, Context())
    assert type(drawn) is Result and type(written) is Result
    assert isinstance(drawn.output, Picture) and isinstance(written.output, str)
    assert drawn.binding is Binding.NEGATION and written.binding is Binding.NEGATION
    assert str(drawn.unsigned) == 'x\n-\ny' and written.unsigned == r'\frac{x}{y}'


def test_one_scale_of_bindings() -> None:
    scale = [Binding.LOWEST, Binding.BLOCK, Binding.IMPLICATION, Binding.XOR, Binding.OR, Binding.AND,
             Binding.RELATION, Binding.MODULO, Binding.SET_OPERATION, Binding.SUM, Binding.NEGATION, Binding.QUOTIENT,
             Binding.PRODUCT, Binding.FRACTION, Binding.ROOT, Binding.POWER, Binding.AFFIX, Binding.APPLICATION,
             Binding.NOT, Binding.ATOM]
    assert scale == sorted(Binding) and len(set(scale)) == len(Binding)
    # each printer says how its own notation binds: a quotient is a
    # built-up \frac in LaTeX and a fraction bar in a picture
    pictures = PrettyPrinter(PrettySettings(), PRETTY_RULES)
    strings = LatexPrinter(LatexSettings())
    assert pictures.rule(x/y, TOP).binding is Binding.FRACTION and pictures.rule(x/y, TOP).stacked
    assert strings.rule(x/y, Context()).binding is Binding.PRODUCT and not strings.rule(x/y, Context()).stacked
    for e in (x + y, -x, x*y, x**y, sin(x), x, x < y):
        assert pictures.rule(e, TOP).binding is strings.rule(e, Context(inline=True)).binding, e


def test_one_place_for_the_brackets() -> None:
    top: Context[str] = Context()
    # the flags the pretty printer brought into the framework
    bar = Result('a over b', Binding.FRACTION, stacked=True)
    assert needs_brackets(bar, top.numerator(False)) and needs_brackets(bar, top.denominator(False))
    assert not needs_brackets(bar, top.term()) and not needs_brackets(bar, top.delimited())
    assert not needs_brackets(Result('a over b', Binding.FRACTION), top.numerator(False))
    raised = top.at(Position.EXPONENT, Binding.LOWEST, excluded=Binding.POWER)
    assert needs_brackets(Result('y^z', Binding.POWER), raised)
    assert not needs_brackets(Result('y + z', Binding.SUM), raised)
    assert raised.in_exponent and not top.at(Position.BASE, Binding.POWER).in_exponent
    assert top.displayed().display and not top.displayed().term().display
    # enclose is the second half of print
    printer = PrettyPrinter(PrettySettings(), PRETTY_RULES)
    c = TOP.factor(followed=True)
    assert printer.enclose(printer.rule(x + y, c), c) == printer.print(x + y, c)
    assert str(printer.print(x + y, c).output) == '(x + y)' and printer.print(x + y, c).binding is Binding.ATOM


def test_a_function_takes_the_exponent_its_parent_offers_in_both_printers() -> None:
    offered = TOP.at(Position.BASE, Binding.POWER, strict=True, exponent_offered=Picture.text('2'))
    printer = PrettyPrinter(PrettySettings(), PRETTY_RULES)
    result = printer.rule(sin(x), offered)
    assert result.absorbed and result.binding is Binding.POWER and str(result.output) == '   2   \nsin (x)'
    assert not printer.rule(x, offered).absorbed
    strings = LatexPrinter(LatexSettings())
    written = strings.rule(sin(x), Context[str]().base('2'))
    assert written.absorbed and written.binding is Binding.POWER and written.output == r'\sin^{2}{\left(x \right)}'
    assert pretty(sin(x)**2) == '   2   \nsin (x)' and latex(sin(x)**2) == r'\sin^{2}{\left(x \right)}'
    # a tall exponent is not offered: the application is bracketed (SymPy's layout)
    assert pretty(Pow(sin(x), y/2)).split('\n')[-1].startswith('(sin(x))')


def test_a_connective_takes_the_negation_its_parent_offers_in_both_printers() -> None:
    assert pretty(Not(Implies(x, y)), use_unicode=True) == 'x ↛ y'
    assert pretty(Not(Equivalent(x, y)), use_unicode=True) == 'x ⇎ y'
    assert pretty(Not(And(x, y)), use_unicode=True) == '¬(x ∧ y)'
    assert pretty(Not(Implies(x, y))) == 'Not(Implies(x, y))'
    assert latex(Not(Implies(x, y))) == r'x \not\Rightarrow y'
    printer = PrettyPrinter(PrettySettings(use_unicode=True), PRETTY_RULES)
    negated = TOP.at(Position.NEGATED, Binding.AFFIX, negation_offered=True)
    assert printer.rule(Implies(x, y), negated).absorbed and not printer.rule(And(x, y), negated).absorbed
    assert not printer.rule(Not(Implies(x, y)), TOP).absorbed


# ----------------------------------------------------------------------
# shared conventions

def test_the_names_are_read_in_one_place() -> None:
    assert _latex_names.split_super_sub is conventions.split_super_sub
    assert symbology.split_super_sub is conventions.split_super_sub
    assert conventions.split_super_sub('alpha_1^2') == ('alpha', ['2'], ['1'])
    assert latex(Symbol('alphahat_1')) == r'\hat{\alpha}_{1}'
    assert pretty(Symbol('alphahat_1'), use_unicode=True) == sympy.pretty(sympy.Symbol('alphahat_1'), use_unicode=True)
    assert 'lambda' in conventions.GREEK_LETTERS and 'lamda' not in conventions.GREEK_LETTERS
    assert conventions.GREEK_LETTERS == frozenset(conventions.GREEK_NAMES) and len(conventions.GREEK_NAMES) == 24
    assert all(name in symbology.greek_unicode for name in conventions.GREEK_NAMES)
    assert latex(Symbol('lamda')) == r'\lambda' and pretty(Symbol('lamda'), use_unicode=True) == 'λ'


def test_the_terms_are_ordered_in_one_place() -> None:
    # the printers and sstr write the terms in the same order, SymPy's
    for e in (exp(-2*z/3) + cos(x), x + y**2 + x*y + 1, 1 - x, x**2 - x - 1):
        expected = sympy.sympify(str(e))
        assert str(e) == str(expected)
        assert latex(e) == sympy.latex(expected)
        assert pretty(e) == sympy.pretty(expected, use_unicode=False)
    e = x + y**2 + x*y + 1
    assert latex(e, order='grlex') == 'x y + y^{2} + x + 1'
    unevaluated = Mul(y, x, Integer(2))
    assert pretty(unevaluated, order='none') == 'y*x*2' and latex(unevaluated, order='none') == r'y x \cdot 2'
    assert pretty(Add(Integer(1), x), order='none') == '1 + x' and latex(Add(Integer(1), x), order='none') == '1 + x'


# ----------------------------------------------------------------------
# the items which were left open

def test_ite_is_drawn_as_its_cases() -> None:
    sx, sy, sz = sympy.symbols('x y z')
    for unicode in (True, False):
        assert pretty(ITE(x, y, z), use_unicode=unicode) == \
            sympy.pretty(sympy.ITE(sx, sy, sz, evaluate=False), use_unicode=unicode)
        assert pretty(And(ITE(x, y, z), y), use_unicode=unicode) == \
            sympy.pretty(sympy.And(sympy.ITE(sx, sy, sz, evaluate=False), sy), use_unicode=unicode)


def test_full_prec_of_the_pretty_printer() -> None:
    half = Float('0.5')
    sympy_half, sx = sympy.Float('0.5'), sympy.Symbol('x')
    assert pretty(half) == sympy.pretty(sympy_half) == '0.500000000000000'
    assert pretty(half + x) == sympy.pretty(sympy_half + sx, use_unicode=False) == 'x + 0.5'
    assert pretty(half + x, full_prec=True) == sympy.pretty(sympy_half + sx, full_prec=True, use_unicode=False) \
        == 'x + 0.500000000000000'
    assert pretty(half, full_prec=False) == sympy.pretty(sympy_half, full_prec=False) == '0.5'
    out = io.StringIO()
    with redirect_stdout(out):
        pprint(half*x, full_prec=True)
    assert out.getvalue() == '0.500000000000000*x\n'


# ----------------------------------------------------------------------
# the interactive session

def test_init_printing_plugs_the_pretty_printer_in() -> None:
    previous_unicode = pretty_use_unicode()
    try:
        assert pretty_printer() is None and text_form(x**2/2) == 'x**2/2' and _shown(x**2/2) == 'x**2/2'
        init_printing(display_hook=False)
        assert pretty_printer() is native_pretty
        assert text_form(x**2/2) == pretty(x**2/2, use_unicode=True) == ' 2\nx \n──\n2 '
        # a picture of several rows starts on a new line
        assert _shown(x**2/2) == '\n' + pretty(x**2/2, use_unicode=True) and _shown(x + 1) == 'x + 1'
        assert pretty(1/x) == pretty(1/x, use_unicode=True)  # the default of pretty follows
        init_printing(use_unicode=False, display_hook=False)
        assert text_form(x**2/2) == ' 2\nx \n--\n2 ' and pretty(1/x) == pretty(1/x, use_unicode=False)
        # the matrices, the polynomials and the Python containers
        assert text_form(ExprMatrix.from_rows([[x, Integer(1)]])) == '[x  1]'
        assert text_form(Poly(x + 1, x)) == "Poly(x + 1, x, domain='ZZ')"
        assert text_form([x, Fraction(1, 2), {x: (y,)}]) == '[x, 1/2, {x: (y,)}]'
        init_printing(pretty_print=False, display_hook=False)
        assert text_form(x**2/2) == 'x**2/2' and _shown(x**2/2) == 'x**2/2'
        # the settings shared with the LaTeX printer
        init_printing(order='none', root_notation=False, full_prec=True, use_unicode=False, display_hook=False)
        assert text_form(Mul(y, x, Integer(2))) == 'y*x*2'
        assert text_form(Pow(x, Rational(1, 2))) == pretty(Pow(x, Rational(1, 2)), root_notation=False)
        assert text_form(Float('0.5')*x) == '0.500000000000000*x'
        sink = _Sink()
        repr_pretty(x, sink, True)
        assert sink.written == ['...']
    finally:
        reset_printing()
    assert pretty_printer() is None and text_form(x**2/2) == 'x**2/2' and pretty_use_unicode() == previous_unicode


def test_a_printer_plugged_in_by_the_user_is_kept() -> None:
    def shout(e: LatexInput, use_unicode: bool) -> str:
        return str(e).upper()
    try:
        set_pretty_printer(shout)
        init_printing(display_hook=False)
        assert pretty_printer() is shout and text_form(x + 1) == 'X + 1' and _shown(x + 1) == 'X + 1'
        reset_printing()
        assert pretty_printer() is shout
    finally:
        set_pretty_printer(None)
        reset_printing()


def test_the_display_hook_draws_the_results() -> None:
    previous = sys.displayhook
    out = io.StringIO()
    try:
        init_printing(display_hook=True, use_unicode=False)
        with redirect_stdout(out):
            sys.displayhook(1/x)
            sys.displayhook(3)
    finally:
        reset_printing()
    assert out.getvalue() == '1\n-\nx\n3\n' and sys.displayhook is previous


def test_domain_matrix_in_a_notebook() -> None:
    m = DomainMatrix([[Fraction(1), Fraction(1, 2)], [Fraction(3), Fraction(4)]], QQ)
    assert m._repr_latex_() == r'$\displaystyle \left[\begin{matrix}1 & \frac{1}{2}\\3 & 4\end{matrix}\right]$'
    empty: DomainMatrix[int] = DomainMatrix([], ZZ, 2)
    assert empty._repr_latex_() is None
    sink = _Sink()
    m._repr_pretty_(sink, False)
    assert sink.written == ['DomainMatrix([[1, 1/2], [3, 4]], QQ)']
    try:
        init_printing(display_hook=False, use_unicode=False)
        sink = _Sink()
        m._repr_pretty_(sink, False)
        assert sink.written == ['\n[1  1/2]\n[      ]\n[3   4 ]']
        init_printing(use_latex=False, display_hook=False)
        assert m._repr_latex_() is None
    finally:
        reset_printing()
