"""The pretty printer: its settings, the objects which are not nodes,
the pictures, the symbology and the coverage of the node classes."""
from __future__ import annotations

import pytest

from sympy_extras.assumptions.quantifiers import Exists, ForAll
from sympy_extras.concrete.expr_with_limits import ExprWithLimits
from sympy_extras.core import Add, And, I, Integer, Pow, Q, Rational, Symbol, expand, oo, pi, sqrt
from sympy_extras.core.basic import Atom, Basic
from sympy_extras.core.expr import AtomicExpr, Expr
from sympy_extras.core.function import Function
from sympy_extras.core.logic import Boolean
from sympy_extras.core.numbers import Number
from sympy_extras.core.operations import AssocOp, LatticeOp
from sympy_extras.core.symbol import Dummy, Wild
from sympy_extras.functions.elementary.trigonometric import cos, sin
from sympy_extras.integrals.integral import Integral
from sympy_extras.integrals.regions import IntegralByRanges
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys.complexrootof import ComplexRootOf
from sympy_extras.polys.polytools import Poly
from sympy_extras.printing.pretty import (RULES, TOP, Binding, Context, Picture, Position, PrettyPrinter,
    PrettyResult, PrettySettings, Symbology, needs_brackets, pprint, pretty, pretty_picture, pretty_repr,
    pretty_use_unicode)
from sympy_extras.printing.pretty.symbology import line_width, split_super_sub
from sympy_extras.series.formal import fps
from sympy_extras.series.fourier import fourier_series
from sympy_extras.series.sequences import SeqBase, SeqFormula, SeqPer
from sympy_extras.series.series_class import SeriesBase
from sympy_extras.sets.sets import Set
from sympy_extras.solvers.isolation import TranscendentalRoot

x, y, z, n = (Symbol(name) for name in 'xyzn')


def _u(*rows: str) -> str:
    return '\n'.join(rows)


# ----------------------------------------------------------------------
# the settings

def test_unicode_and_ascii() -> None:
    assert pretty(pi*x**2) == _u('    2', 'pi*x ')
    assert pretty(pi*x**2, use_unicode=True) == _u('   2', 'π⋅x ')
    assert pretty(x <= y) == 'x <= y' and pretty(x <= y, use_unicode=True) == 'x ≤ y'


def test_the_default_of_use_unicode() -> None:
    previous = pretty_use_unicode()
    try:
        assert pretty_use_unicode(True) == previous
        assert pretty(pi) == 'π'
        assert pretty(pi, use_unicode=False) == 'pi'
        pretty_use_unicode(False)
        assert pretty(pi) == 'pi'
    finally:
        pretty_use_unicode(previous)


def test_wrap_line_and_num_columns() -> None:
    e = expand((x + y)**7)
    wrapped = pretty(e, num_columns=40)
    assert wrapped == _u(' 7      6         5  2       4  3      >',
                         'x  + 7*x *y + 21*x *y  + 35*x *y  + 35 >',
                         '',
                         '>   3  4       2  5        6    7',
                         '> *x *y  + 21*x *y  + 7*x*y  + y ')
    assert all(len(row) <= 40 for row in wrapped.split('\n'))
    assert pretty(e, num_columns=40, use_unicode=True).split('\n')[0].endswith(' ↪')
    assert len(pretty(e, num_columns=40, wrap_line=False).split('\n')) == 2
    assert pretty(e, num_columns=200) == pretty(e, wrap_line=False)


def test_order() -> None:
    # 'none' keeps the order of the arguments, the default is the printing order
    e = Add(Integer(1), x, x**2)
    assert pretty(e) == _u(' 2        ', 'x  + x + 1')
    assert pretty(e, order='none') == _u('         2', '1 + x + x ')
    with pytest.raises(ValueError):
        pretty(e, order='grevlex')


def test_root_notation() -> None:
    assert pretty(x**Rational(1, 3)) == _u('3 ___', '\\/ x ')
    assert pretty(x**Rational(1, 3), root_notation=False) == _u(' 1/3', 'x   ')
    assert pretty(sqrt(x), use_unicode=True) == '√x'
    assert pretty(sqrt(x), use_unicode=True, use_unicode_sqrt_char=False) == _u('  ___', '╲╱ x ')
    assert pretty(Pow(x, 1/n), use_unicode=True) == _u('n ___', '╲╱ x ')


def test_imaginary_unit() -> None:
    assert pretty(I*x + 1, use_unicode=True) == 'ⅈ⋅x + 1'
    assert pretty(I*x + 1, use_unicode=True, imaginary_unit='j') == 'ⅉ⋅x + 1'
    assert pretty(I*x + 1) == 'I*x + 1'
    with pytest.raises(ValueError):
        pretty(I, imaginary_unit='k')


def test_a_float_alone_has_all_its_digits() -> None:
    from sympy_extras.core import Float
    assert pretty(Float(2.5)) == '2.50000000000000'
    assert pretty(Float(2.5)*x) == '2.5*x'


# ----------------------------------------------------------------------
# known pictures (SymPy's tutorial on printing)

def test_known_integral_of_a_root() -> None:
    e = Integral(Pow(Pow(x, Integer(-1)), Rational(1, 2)), x)
    assert pretty(e, use_unicode=True) == _u('⌠           ',
                                             '⎮     ___   ',
                                             '⎮    ╱ 1    ',
                                             '⎮   ╱  ─  dx',
                                             '⎮ ╲╱   x    ',
                                             '⌡           ')
    assert pretty(e) == _u('  /          ',
                           ' |           ',
                           ' |     ___   ',
                           ' |    / 1    ',
                           ' |   /  -  dx',
                           ' | \\/   x    ',
                           ' |           ',
                           '/            ')


def test_known_fraction_and_power() -> None:
    assert pretty((x**2 + 1)/(y - 1)) == _u(' 2    ', 'x  + 1', '------', 'y - 1 ')
    assert pretty(1/(1 + 1/x), use_unicode=True) == _u('  1  ', '─────', '    1', '1 + ─', '    x')


# ----------------------------------------------------------------------
# what is not a node

def test_python_containers() -> None:
    assert pretty([x, 1, (x**2,), ()]) == _u('         2       ', '[x, 1, (x ,), ()]')
    assert pretty({'a': [x, 1], (x, 2): x**2}) == _u('                     2 ', '{a: [x, 1], (x, 2): x }')
    assert pretty({x, y}) == '{x, y}' and pretty(set()) == 'set()'
    assert pretty(frozenset([x])) == 'frozenset({x})' and pretty(frozenset()) == 'frozenset()'
    assert pretty(-3) == '-3' and pretty(None) == 'None' and pretty(True) == 'True' and pretty('text') == 'text'


def test_a_matrix() -> None:
    m = ExprMatrix.from_rows([[1, x], [y**2, 1/z]])
    assert pretty(m, use_unicode=True) == _u('⎡1   x⎤', '⎢     ⎥', '⎢ 2  1⎥', '⎢y   ─⎥', '⎣    z⎦')
    assert pretty(ExprMatrix.column([x, y])) == _u('[x]', '[ ]', '[y]')
    assert pretty([m]).count('\n') == 4


def test_a_polynomial() -> None:
    assert pretty(Poly(x**2 + 1, x)) == _u('    / 2                    \\', "Poly\\x  + 1, x, domain='ZZ'/")
    assert pretty(Poly(x + 1, x), use_unicode=True) == "Poly(x + 1, x, domain='ZZ')"


def test_pprint_prints(capsys: pytest.CaptureFixture[str]) -> None:
    pprint(x**2, use_unicode=False)
    assert capsys.readouterr().out == ' 2\nx \n'


class _Sink:
    def __init__(self) -> None:
        self.written: list[str] = []

    def text(self, s: str) -> None:
        self.written.append(s)


def test_pretty_repr() -> None:
    # what init_printing installs as _repr_pretty_: a tall picture starts on a new line
    sink = _Sink()
    pretty_repr(x + 1, sink, False, PrettySettings())
    pretty_repr(x**2, sink, False, PrettySettings(use_unicode=True))
    pretty_repr(x**2, sink, True)
    assert sink.written == ['x + 1', '\n 2\nx ', '...']


# ----------------------------------------------------------------------
# pictures

def test_pictures() -> None:
    half = Picture.text('1').below(Picture.text('-'), Picture.text('2')).with_baseline(1)
    row = Picture.beside(Picture.text('10'), Picture.text(' + '), half)
    assert row.rows == ('     1', '10 + -', '     2') and row.baseline == 1
    assert row.height() == 3 and row.width() == 6
    assert Picture.text('a').above(Picture.text('bcd')).rows == ('bcd', ' a ')
    assert Picture.text('a').above(Picture.text('bcd')).baseline == 1
    assert Picture.text('ab\nc').rows == ('ab', 'c ')
    ascii, unicode = Symbology(False), Symbology(True)
    assert half.parens(ascii).rows == ('/1\\', '|-|', '\\2/')
    assert half.parens(unicode, '[', ']').rows == ('⎡1⎤', '⎢-⎥', '⎣2⎦')
    assert half.parens(ascii, '{', '}', one_row_in_ascii=True).rows == (' 1 ', '{-}', ' 2 ')
    assert half == Picture(['1', '-', '2'], 1) and half != Picture(['1', '-', '2'], 0)
    assert pretty_picture(x/y, PrettySettings()).baseline == 1


def test_the_binding_decides_the_brackets() -> None:
    printer = PrettyPrinter(PrettySettings(), RULES)
    assert printer.result(-x*y, TOP).binding is Binding.NEG
    assert printer.result(x/y, TOP).binding is Binding.DIV and printer.result(x/y, TOP).stacked
    assert printer.result(Rational(1, 2), TOP).binding is Binding.INLINE_QUOTIENT
    assert printer.result(sin(x), TOP).binding is Binding.FUNC
    assert printer.result(sin(x)**2, TOP).binding is Binding.POW
    assert printer.result(x, TOP).binding is Binding.ATOM
    base = Context(Position.BASE, Binding.POW, strict=True)
    assert needs_brackets(PrettyResult(Picture.text('x**y'), Binding.POW), base)
    assert not needs_brackets(PrettyResult(Picture.text('f(x)'), Binding.FUNC), base)
    # an operand which is open to the right is bracketed when something follows it
    factor = Context(Position.INFIX_FIRST, Binding.MUL, closed=True)
    assert needs_brackets(PrettyResult(Picture.text('lim x'), Binding.MUL, open_right=True), factor)
    assert not needs_brackets(PrettyResult(Picture.text('lim x'), Binding.MUL, open_right=True),
                              Context(Position.INFIX_LAST, Binding.MUL))
    # the contexts are immutable and count the depth
    child = TOP.child(Position.EXPONENT, Binding.OPEN)
    assert child.depth == 1 and child.in_exponent and child.argument().in_exponent and not TOP.in_exponent


# ----------------------------------------------------------------------
# symbology

def test_symbol_names() -> None:
    unicode, ascii = Symbology(True), Symbology(False)
    assert unicode.symbol('alpha') == 'α' and ascii.symbol('alpha') == 'alpha'
    assert unicode.symbol('x_1') == 'x₁' and unicode.symbol('x1') == 'x₁' and unicode.symbol('x_1^2') == 'x²₁'
    assert unicode.symbol('Lambda_beta') == 'Λᵦ' and unicode.symbol('x_y') == 'x_y'
    assert line_width(unicode.symbol('xdot')) == 1 and unicode.symbol('xprime') == 'x′'
    assert split_super_sub('a_x^1') == ('a', ['1'], ['x'])
    assert pretty(Symbol('theta_0') + 1, use_unicode=True) == 'θ₀ + 1'
    assert pretty(Dummy('t'), use_unicode=True) == 't' and pretty(Wild('w')) == 'w_'


def test_extensible_objects() -> None:
    unicode, ascii = Symbology(True), Symbology(False)
    assert ascii.vobj('(', 3) == '/\n|\n\\' and ascii.vobj('(', 1) == '('
    assert unicode.xobj('{', 4) == ['⎧', '⎪', '⎨', '⎪', '⎩']
    assert unicode.hobj('-', 3) == '───' and ascii.hobj('-', 3) == '---'
    assert unicode.xsym('*') == '⋅' and ascii.xsym('!=') == '!='
    with pytest.raises(ValueError):
        ascii.xobj('(', 0)


# ----------------------------------------------------------------------
# the other nodes

def test_quantified_formulas() -> None:
    assert pretty(ForAll(x, Exists(y, x < y)), use_unicode=True) == '∀x ∃y (x < y)'
    assert pretty(ForAll((x, y), And(x < y, y < 1)), use_unicode=True) == '∀x, y (x < y ∧ y < 1)'
    assert pretty(ForAll(x, x < 1)) == 'ForAll(x, x < 1)'
    assert pretty(Q.positive(x)) == 'Q.positive(x)'


def test_sequences_and_series() -> None:
    assert pretty(SeqFormula(n**2, (n, 0, oo)), use_unicode=True) == '[0, 1, 4, 9, …]'
    assert pretty(SeqFormula(n**2, (n, 0, 3))) == '[0, 1, 4, 9]'
    assert pretty(SeqPer((1, 2, 3), (n, 0, oo))) == '[1, 2, 3, 1, ...]'
    # symbolic bounds: SymPy raises NotImplementedError
    assert pretty(SeqFormula(n, (n, 0, x))) == 'SeqFormula(n, (n, 0, x))'
    series = pretty(fourier_series(x, (x, -pi, pi)))
    assert series.split('\n')[1] == '2*sin(x) - sin(2*x) + ---------- + ...'
    assert 'oo' in pretty(fps(cos(x), x))


def test_roots() -> None:
    assert pretty(ComplexRootOf(x**3 - x - 1, 0), use_unicode=True) == _u('       ⎛ 3           ⎞',
                                                                          'CRootOf⎝x  - x - 1, 0⎠')


def test_every_node_class_is_printed_by_a_rule() -> None:
    # every class of nodes has a rule, except the abstract ones and those
    # printed in their constructor form on purpose
    generic: set[type[Basic]] = {Atom, AtomicExpr, Expr, Boolean, Number, AssocOp, LatticeOp, Set, SeriesBase,
                                 ExprWithLimits, TranscendentalRoot, IntegralByRanges}
    printer = PrettyPrinter(PrettySettings(), RULES)
    missing: list[str] = []
    pending: list[type[Basic]] = [Basic]
    seen: set[type[Basic]] = set()
    while pending:
        cls = pending.pop()
        for sub in cls.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            pending.append(sub)
            if not printer.has_rule(sub) and sub not in generic and not sub.__name__.startswith('_') \
                    and not sub.__module__.startswith('sympy_extras.interop'):
                missing.append(sub.__name__)
    assert len(seen) > 250
    assert missing == []
    assert issubclass(SeqFormula, SeqBase) and printer.has_rule(SeqFormula)


def test_a_node_without_a_rule_is_printed_in_its_string_form() -> None:
    printer = PrettyPrinter(PrettySettings(), RULES[:3])
    f = Function('f')
    result = printer.result(f(x, x), TOP)
    assert str(result.picture) == 'f(x, x)' and result.binding is Binding.FUNC
    # a string of unknown structure is bracketed everywhere
    assert printer.opaque('a + b').binding is Binding.OPEN and printer.opaque('ab').binding is Binding.ATOM
    assert printer.opaque('f(x) + g(y)').binding is Binding.OPEN and printer.opaque('f((x), y)').binding is Binding.FUNC
    rows = pretty(Pow(TranscendentalRoot(cos(x) - x, x, 0, 1), Integer(2))).split('\n')
    assert rows[0].strip() == '2' and rows[1] == 'TranscendentalRoot(-x + cos(x), x, 0, 1) '
