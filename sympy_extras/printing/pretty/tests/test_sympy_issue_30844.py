"""The defects of SymPy's pretty printer listed in sympy/sympy#30844 are
not reproduced: one test per root cause and per item of the pretty
section which concerns a class of this package. Each test quotes what
SymPy prints.
"""
from __future__ import annotations

from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.core import Add, Float, Integer, Mul, Not, Pow, Rational, Symbol, oo
from sympy_extras.core.function import Function
from sympy_extras.core.limits import Limit
from sympy_extras.core.mod import Mod
from sympy_extras.core.relational import Lt
from sympy_extras.functions.combinatorial.numbers import DivisorSigma, ReducedTotient
from sympy_extras.functions.elementary.exponential import Exp, ExpPolar
from sympy_extras.functions.elementary.trigonometric import Sin
from sympy_extras.functions.special.singularity_functions import SingularityFunctionNode
from sympy_extras.functions.special.zeta_functions import RiemannXi
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.printing.pretty import (TOP, Binding, Picture, PrettyPrinter, PrettyResult, PrettySettings, RULES,
    needs_brackets, pretty)
from sympy_extras.printing.pretty.tests.test_sympy_agree import _ascii_only, _corpus, _unicode_only
from sympy_extras.sets.sets import Complement, Contains, FiniteSet, Interval, ProductSet, Union

x, y, z, n, k, a, b, t = (Symbol(name) for name in 'xyznkabt')
f = Function('f')
X, Y, Z = Interval(0, 1), Interval(2, 3), FiniteSet(5)


def _u(*rows: str) -> str:
    return '\n'.join(rows)


# ----------------------------------------------------------------------
# root causes

def test_root_cause_1_a_negative_product_is_not_a_product() -> None:
    # SymPy: precedence_traditional(-x) is that of a product, so that
    # Mod(-x, y) and -Mod(x, y) both print "-x mod y"
    assert pretty(Mod(-x, y)) == '(-x) mod y'
    assert pretty(Mul(Integer(-1), Mod(x, y))) == '-(x mod y)'
    # a negative factor after the first one: SymPy prints x*-2 for Mul(x, -1, 2)
    assert pretty(Mul(x, Integer(-1), Integer(2)), use_unicode=True) == 'x⋅(-1)⋅2'
    assert pretty(Mul(x, Mul(Integer(-1), y)), use_unicode=True) == 'x⋅(-y)'


def test_root_cause_3_the_sides_of_a_relation_are_bracketed() -> None:
    # SymPy: Eq(Eq(x, y), Eq(y, z)) prints "x = y = y = z", Eq(And(p, q), r) "p ∧ q = r".
    # The sides of a relation of this package are expressions, so the
    # trees cannot be built; the rule of the brackets is checked itself.
    side = TOP.side()
    assert needs_brackets(PrettyResult(Picture.text('x = y'), Binding.RELATION), side)
    assert needs_brackets(PrettyResult(Picture.text('p ∧ q'), Binding.AND), side)
    assert not needs_brackets(PrettyResult(Picture.text('x + y'), Binding.SUM), side)
    # a membership is a relation too: its element is a side
    assert pretty(Contains(x + 1, X), use_unicode=True) == 'x + 1 ∈ [0, 1]'
    assert pretty(Not(Contains(x, X)), use_unicode=True) == '¬(x ∈ [0, 1])'


def test_root_cause_4_not_brackets_relations() -> None:
    # SymPy tests arg.is_Boolean, which is False for a relation: "¬t < x", "¬x ∈ [0, 1]"
    assert pretty(Not(Lt(t, x)), use_unicode=True) == '¬(t < x)'
    assert pretty(Not(Contains(x, X)), use_unicode=True) == '¬(x ∈ [0, 1])'
    assert pretty(Not(Not(a)), use_unicode=True) == '¬¬a'
    assert pretty(Not(a), use_unicode=True) == '¬a'


def test_root_cause_5_composite_pictures_are_not_atoms() -> None:
    # SymPy: Pow(Rational(2, 3), x) prints 2/3 with the exponent x (read 2/3**x)
    assert pretty(Pow(Rational(2, 3), x)) == _u('     x', '(2/3) ')
    # SymPy: Rational(-1, 2)**n + x prints -1/2**n + x
    assert pretty(Add(Pow(Rational(-1, 2), n), x)) == _u('      n    ', '(-1/2)  + x')
    # SymPy: Float(-2.5)**n prints -2.5**n
    assert pretty(Pow(Float(-2.5), n)) == _u('      n', '(-2.5) ')
    assert pretty(Pow(Integer(-2), n)) == _u('    n', '(-2) ')
    # in a product: (1/2)*3 is not 1/(2*3)
    assert pretty(Mul(Rational(1, 2), Integer(3))) == '(1/2)*3'


def test_root_cause_5_no_composite_result_is_an_atom() -> None:
    # by construction: the binding of a result is the one of the rule
    # which made the picture. A one-row atom is one token or is closed
    # by its own delimiters.
    opening, closing = '([{|<⌊⌈√', ')]}|>⌋⌉'
    for unicode in (True, False):
        printer = PrettyPrinter(PrettySettings(use_unicode=unicode), RULES)
        for e in _corpus() + _unicode_only() + _ascii_only():
            result = printer.rule(e, TOP)
            if result.binding is not Binding.ATOM or result.output.height() != 1:
                continue
            text = str(result.output)
            token = all(c.isalnum() or c in '._^' or ord(c) > 127 for c in text) and ' ' not in text
            closed = text[0] in opening and text[-1] in closing
            assert token or closed, (e, text)


def test_root_cause_6_the_direction_mark_applies_to_the_point() -> None:
    # SymPy: Limit(f(x), x, a + b) prints "x─→a + b⁺"
    assert pretty(Limit(f(x), x, a + b), use_unicode=True) == _u('    lim    f(x)', 'x─→(a + b)⁺    ')
    assert pretty(Limit(x, x, y + z, '-')) == _u('    lim    x', 'x->(y + z)- ')
    # nothing to disambiguate without a mark or for an atom
    assert pretty(Limit(f(x), x, a + b, '+-')) == _u('  lim   f(x)', 'x->a + b    ')
    assert pretty(Limit(f(x), x, a)) == _u(' lim f(x)', 'x->a+    ')


# ----------------------------------------------------------------------
# the pretty section: the picture reads as another expression

def test_a_sum_sign_does_not_capture_the_following_factors() -> None:
    # SymPy puts the Sum inside the body of the Product in Product(k, (k, 1, n))*Sum(1/k, (k, 1, n))
    product, total = Product(k, (k, 1, n)), Sum(1/k, (k, 1, n))
    first = pretty(Mul(product, total), use_unicode=True, order='none')
    assert first == _u('           n     ',
                       '          ____   ',
                       '          ╲      ',
                       '⎛  n    ⎞  ╲     ',
                       '⎜─┬─┬─  ⎟   ╲   1',
                       '⎜ │ │  k⎟⋅  ╱   ─',
                       '⎜ │ │   ⎟  ╱    k',
                       '⎝k = 1  ⎠ ╱      ',
                       '          ‾‾‾‾   ',
                       '          k = 1  ')
    # the last factor needs no brackets, any other one has them
    for expression in (Mul(total, product), Mul(total, x), Mul(total, Limit(f(x), x, 0))):
        picture = pretty(expression, use_unicode=True, order='none')
        assert picture.splitlines()[0].startswith('⎛'), picture
    assert '⎛' not in pretty(Mul(x, total), use_unicode=True, order='none')


def test_mod_brackets_its_operands() -> None:
    # SymPy: "x mod y + z", "x mod y mod z" (sympy/sympy#23912)
    assert pretty(Mod(x, y + z)) == 'x mod (y + z)'
    assert pretty(Mod(x, Mod(y, z))) == 'x mod (y mod z)'
    assert pretty(Mod(Mod(x, y), z)) == '(x mod y) mod z'
    assert pretty(Add(Mod(x, y), z), order='none') == '(x mod y) + z'
    assert pretty(Mul(Integer(2), Mod(x, y))) == '2*(x mod y)'
    assert pretty(Pow(Mod(x, y), Integer(2))) == _u('         2', '(x mod y) ')


def test_set_operations_bracket_their_operands() -> None:
    # SymPy: Complement(X, Complement(Y, Z)) prints like (X \ Y) \ Z
    assert pretty(Complement(X, Complement(Y, Z))) == '[0, 1] \\ ([2, 3] \\ {5})'
    assert pretty(Complement(Complement(X, Y), Z)) == '([0, 1] \\ [2, 3]) \\ {5}'
    # SymPy: ProductSet(Complement(X, Y), Z) prints "X \ Y × Z"
    assert pretty(ProductSet(Complement(X, Y), Z), use_unicode=True) == '([0, 1] \\ [2, 3]) × {5}'
    # SymPy: ProductSet(U, U) with U = Union(X, Y) prints "X ∪ Y²"
    assert pretty(ProductSet(Union(X, Y), Union(X, Y)), use_unicode=True) == _u('                 2',
                                                                               '([0, 1] ∪ [2, 3]) ')


def test_a_power_of_a_function_power_is_bracketed() -> None:
    # SymPy: Pow(sin(x)**2, 3) prints sin²(x) with a stray 3 above
    assert pretty(Pow(Pow(Sin(x), Integer(2)), Integer(3)), use_unicode=True) == _u('         3',
                                                                                   '⎛   2   ⎞ ',
                                                                                   '⎝sin (x)⎠ ')
    assert pretty(Pow(Sin(x), Integer(2))) == _u('   2   ', 'sin (x)')


def test_the_operand_of_a_limit_does_not_touch_the_point() -> None:
    # SymPy: Limit(1/x, x, oo) prints the denominator next to the point: "x─→∞x"
    assert pretty(Limit(1/x, x, oo), use_unicode=True) == _u('    ⎛1⎞', 'lim ⎜─⎟', 'x─→∞⎝x⎠')
    # an operand which is not bracketed and has rows under its baseline is kept apart
    assert pretty(Limit(f(1/x), x, oo), use_unicode=True) == _u('      ⎛1⎞', 'lim  f⎜─⎟', 'x─→∞  ⎝x⎠')


def test_a_product_sign_brackets_a_sum() -> None:
    # SymPy: Product(k + 1, (k, 1, n)) and Product(k, (k, 1, n)) + 1 print the same
    inside = pretty(Product(k + 1, (k, 1, n)), use_unicode=True)
    outside = pretty(Add(Product(k, (k, 1, n)), Integer(1)), use_unicode=True, order='none')
    assert inside != outside
    assert '(k + 1)' in inside and 'k + 1' in outside and '(k + 1)' not in outside


# ----------------------------------------------------------------------
# the pretty section: crashes, settings, lost information

def test_an_empty_matrix_is_printed() -> None:
    # SymPy: pretty(Array([])) raises IndexError
    assert pretty(ExprMatrix.zeros(0, 0)) == '[]'
    assert pretty(ExprMatrix.zeros(0, 3)) == '[]'


def test_the_singularity_function_in_ascii() -> None:
    # SymPy: garbled, "<  a     - - + x   b    >" (it iterates over the rows of a picture)
    assert pretty(SingularityFunctionNode(x, a/b, n)) == _u('         n',
                                                            '/  a    \\ ',
                                                            '|- - + x| ',
                                                            '\\  b    / ')
    assert pretty(SingularityFunctionNode(x, a, n)) == _u('        n', '<-a + x> ')


def test_function_names_are_not_symbol_names() -> None:
    # SymPy: "reducedₜₒₜᵢₑₙₜ(n)", "divisor_σ(n)"
    assert pretty(ReducedTotient(n), use_unicode=True) == 'reduced_totient(n)'
    assert pretty(DivisorSigma(n), use_unicode=True) == 'divisor_sigma(n)'
    assert pretty(RiemannXi(x), use_unicode=True) == 'riemann_xi(x)'
    # an undefined function is named like a symbol
    assert pretty(Function('f_1')(x), use_unicode=True) == 'f₁(x)'
    assert pretty(Function('phi')(x), use_unicode=True) == 'φ(x)'


def test_no_information_is_lost() -> None:
    # SymPy: Symbol('x_') + x prints "x + x"
    assert pretty(Symbol('x_') + x, use_unicode=True) == 'x + x_'
    assert pretty(Symbol('x_'), use_unicode=True) != pretty(x, use_unicode=True)
    # SymPy: exp_polar(x) prints like exp(x)
    assert pretty(ExpPolar(x), use_unicode=True) == 'exp_polar(x)'
    assert pretty(ExpPolar(x), use_unicode=True) != pretty(Exp(x), use_unicode=True)


def test_python_floats_which_are_not_finite() -> None:
    # (second pass) Python's inf and nan were printed as 'inf' and 'nan', and crashed the LaTeX printer: they are
    # the infinity and the nan of the core, as in SymPy
    assert pretty(float('inf'), use_unicode=True) == '∞' and pretty(float('-inf'), use_unicode=True) == '-∞'
    assert pretty(float('inf')) == 'oo' and pretty(float('nan')) == 'nan' and pretty(1.5) == '1.5'
