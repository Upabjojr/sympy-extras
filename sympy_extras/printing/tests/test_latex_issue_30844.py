r"""The defects of SymPy's LaTeX printer listed in sympy/sympy#30844 are
not reproduced: one test per item which concerns a class of this
package, citing the item. The items about classes this package does not
have (matrix expressions, tensors, quantum, units, vectors, control,
arrays, permutations, ring elements) do not apply.
"""
from __future__ import annotations

import sympy

from sympy_extras.core import (And, Derivative, Float, Function, Integer, Mod, Mul, Not, Or, Pow, Rational, Subs,
    Symbol, UnevaluatedExpr, sqrt)
from sympy_extras.core.limits import Limit
from sympy_extras.core.numbers import AlgebraicNumber
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.functions.combinatorial.numbers import (DivisorSigma, JacobiSymbol, UDivisorSigma, divisor_sigma,
    udivisor_sigma)
from sympy_extras.functions.elementary.complexes import Adjoint, atan2
from sympy_extras.functions.elementary.exponential import ExpPolar, exp
from sympy_extras.functions.elementary.hyperbolic import asinh
from sympy_extras.functions.elementary.miscellaneous import Max
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.elementary.trigonometric import Cos, cos, sin
from sympy_extras.functions.special.error_functions import erf2
from sympy_extras.integrals.integral import Integral
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys.polytools import Poly
from sympy_extras.printing import latex
from sympy_extras.sets.conditionset import ConditionSet
from sympy_extras.sets.sets import Contains, Interval, Reals

x, y, z, n, k, a, b = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('n'), Symbol('k'), Symbol('a'), Symbol('b')
f = Function('f')
g = Function('g')


# ---------------------------------------------------------------------
# the root causes

def test_root_cause_1_negative_products_bind_as_negations() -> None:
    # precedence_traditional(-x) is that of a product: Mod(-x, y) and -Mod(x, y) both print - x \bmod y
    assert latex(Mod(-x, y)) == r'\left(- x\right) \bmod y'
    assert latex(Mul(-1, Mod(x, y))) == r'- \left(x \bmod y\right)'
    assert latex(Mod(-x, y)) != latex(Mul(-1, Mod(x, y)))
    reference = sympy.Mod(-sympy.Symbol('x'), sympy.Symbol('y'))
    assert sympy.latex(reference) == sympy.latex(-sympy.Mod(sympy.Symbol('x'), sympy.Symbol('y')))  # the defect


def test_root_cause_3_sides_of_relations_are_bracketed() -> None:
    # Eq(And(p, q), Or(p, q)) prints p \wedge q = p \vee q; the side of a relation brackets what binds
    # no tighter than a relation (the relations of this package take expressions: the rule is checked on
    # the framework, and on a piecewise side, which is bracketed for being a block)
    from sympy_extras.printing.framework import Binding, Context, Result, needs_brackets
    top: Context[str] = Context()
    assert needs_brackets(Result('y = z', Binding.RELATION), top.side())
    assert needs_brackets(Result(r'p \wedge q', Binding.AND), top.side())
    assert not needs_brackets(Result('x + 1', Binding.SUM), top.side())


def test_root_cause_4_not_brackets_relations_and_memberships() -> None:
    # \neg x > y, \neg x \in \left[0, 1\right]
    assert latex(Not(x > y)) == r'\neg \left(x > y\right)'
    assert latex(Not(Contains(x, Interval(0, 1)))) == r'\neg \left(x \in \left[0, 1\right]\right)'
    assert latex(Not(Symbol('p'))) == r'\neg p'
    assert latex(Not(And(x > y, y > z))) == r'\neg \left(x > y \wedge y > z\right)'


def test_root_cause_5_fractions_and_negative_numbers_as_bases() -> None:
    # the LaTeX side of the pretty defect: (2/3)**x, (-1/2)**n, (-2.5)**n keep their brackets
    assert latex(Pow(Rational(2, 3), x)) == r'\left(\frac{2}{3}\right)^{x}'
    assert latex(Pow(Rational(-1, 2), n)) == r'\left(- \frac{1}{2}\right)^{n}'
    assert latex(Pow(Float(-2.5), n)) == r'\left(-2.5\right)^{n}'


def test_root_cause_6_the_direction_of_a_limit_is_a_script_of_the_point() -> None:
    # \lim_{x \to a + b^+}, a^{2}^+ (a double superscript), - a^+
    assert latex(Limit(f(x), x, a + b)) == r'\lim_{x \to \left(a + b\right)^+} f{\left(x \right)}'
    assert latex(Limit(f(x), x, a**2)) == r'\lim_{x \to \left(a^{2}\right)^+} f{\left(x \right)}'
    assert latex(Limit(f(x), x, -a)) == r'\lim_{x \to \left(- a\right)^+} f{\left(x \right)}'
    assert latex(Limit(f(x), x, a, '-')) == r'\lim_{x \to a^-} f{\left(x \right)}'
    assert latex(Limit(f(x), x, 0)) == r'\lim_{x \to 0^+} f{\left(x \right)}'


# ---------------------------------------------------------------------
# crashes and invalid TeX

def test_any_base_can_be_raised_to_a_power() -> None:
    # the exp= keyword: Pow(Piecewise(...), 2) and the functions with a printer of their own raise TypeError
    pw = Piecewise((x, x < 0), (y, True))
    assert latex(Pow(pw, 2)) == \
        r'\left(\begin{cases} x & \text{for}\: x < 0 \\y & \text{otherwise} \end{cases}\right)^{2}'
    assert latex(Pow(Integral(f(x), x), 2)) == r'\left(\int f{\left(x \right)}\, dx\right)^{2}'


def test_no_double_superscripts() -> None:
    # Limit point a**2 (above); adjoint(Symbol('A^a')) -> A^{a}^{\dagger}; udivisor_sigma(n)**2 ->
    # \sigma^*^{2}; UnevaluatedExpr(exp(z))**2 -> e^{z}^{2}
    A = Symbol('A^a', commutative=False)
    assert latex(Adjoint(A)) == r'\left(A^{a}\right)^{\dagger}'
    assert latex(Pow(UDivisorSigma(n), 2)) == r'\left(\sigma^*\left(n\right)\right)^{2}'
    assert latex(udivisor_sigma(n, k)) == r'\sigma^*_{k}\left(n\right)'
    assert latex(Pow(UnevaluatedExpr(exp(z)), 2)) == r'\left(e^{z}\right)^{2}'
    assert latex(Symbol('x^i')**2) == r'\left(x^{i}\right)^{2}'
    assert latex(Symbol('x^i')**2, parenthesize_super=False) == r'{x^{i}}^{2}'


def test_wide_and_empty_matrices() -> None:
    # Array(range(24), (2, 12)): the column specification counted the rows; Array([]): IndexError
    wide = ExprMatrix(2, 12, [Integer(i) for i in range(24)])
    assert r'\begin{array}{cccccccccccc}' in latex(wide)
    assert latex(ExprMatrix(0, 0, [])) == r'\left[\begin{matrix}\end{matrix}\right]'


def test_text_escapes_are_terminated() -> None:
    # latex('a\\b') -> \textbackslashb, an undefined control sequence
    assert latex('a\\b') == r'\mathtt{\text{a\textbackslash{}b}}'
    assert latex('a~b^c') == r'\mathtt{\text{a\textasciitilde{}b\textasciicircum{}c}}'


def test_special_characters_in_names() -> None:
    # Symbol('a%b') + 1 -> a%b + 1 (a comment); Function('a%b')(x) raises ValueError
    assert latex(Symbol('a%b') + 1) == r'a\%b + 1'
    assert latex(Symbol('a#b')) == r'a\#b' and latex(Symbol('a&b')) == r'a\&b' and latex(Symbol('a$b')) == r'a\$b'
    assert latex(Symbol('a{b')) == r'a\{b'
    assert latex(Function('a%b')(x)) == r'\operatorname{a\%b}{\left(x \right)}'
    assert latex(Symbol('x_{1}')) == 'x_{1}'  # LaTeX in a name is kept


def test_undefined_functions_named_like_commands() -> None:
    # Function('frac')(x) -> \frac{\left(x \right)}, Function('root')(x) -> \root{...}
    assert latex(Function('frac')(x)) == r'\operatorname{frac}{\left(x \right)}'
    assert latex(Function('root')(x)) == r'\operatorname{root}{\left(x \right)}'
    assert latex(Function('sin')(x)) == r'\operatorname{sin}{\left(x \right)}'


def test_custom_mul_symbol_stands_between_spaces() -> None:
    # latex(2*x, mul_symbol=r'\star') -> 2\starx
    assert latex(2*x, mul_symbol=r'\star') == r'2 \star x'
    assert latex(2*x*y, mul_symbol=r'\,') == r'2 \, x \, y'


# ---------------------------------------------------------------------
# wrong precedence, wrong value

def test_mod_operands() -> None:
    # Mod(x, -y) -> x \bmod - y; Mod(-x, y)*z -> z \left(- x \bmod y\right);
    # Mod(x, Mod(y, z)) and Mod(Mod(x, y), z) both x \bmod y \bmod z
    assert latex(Mod(x, -y)) == r'x \bmod \left(- y\right)'
    assert latex(Mul(z, Mod(-x, y))) == r'z \left(\left(- x\right) \bmod y\right)'
    assert latex(Mod(x, Mod(y, z))) == r'x \bmod \left(y \bmod z\right)'
    assert latex(Mod(Mod(x, y), z)) == r'\left(x \bmod y\right) \bmod z'
    assert latex(Mod(x + 1, y + 2)) == r'\left(x + 1\right) \bmod \left(y + 2\right)'


def test_subs_under_a_power() -> None:
    # \left. f \right|_{\substack{ x=1 }}^{2} reads as an evaluation between limits
    assert latex(Pow(Subs(f(x), x, 1), 2)) == \
        r'\left(\left. f{\left(x \right)} \right|_{\substack{ x=1 }}\right)^{2}'


def test_operators_followed_by_a_factor() -> None:
    # Derivative(f)*Derivative(g) -> \frac{d}{d x} f \frac{d}{d x} g (reads d/dx[f g']); Limit*Limit, Mul(Limit, 2)
    df, dg = Derivative(f(x), x), Derivative(g(x), x)
    assert latex(Mul(df, dg)) == \
        r'\left(\frac{d}{d x} f{\left(x \right)}\right) \frac{d}{d x} g{\left(x \right)}'
    lim = Limit(f(x), x, 0)
    assert latex(Mul(lim, lim)) == \
        r'\left(\lim_{x \to 0^+} f{\left(x \right)}\right) \lim_{x \to 0^+} f{\left(x \right)}'
    assert latex(Mul(df, 2)) == r'\left(\frac{d}{d x} f{\left(x \right)}\right) \cdot 2'
    assert latex(Mul(lim, 2)) == r'\left(\lim_{x \to 0^+} f{\left(x \right)}\right) \cdot 2'
    assert latex(Mul(2, df)) == r'2 \frac{d}{d x} f{\left(x \right)}'


def test_mod_in_the_body_of_an_operator() -> None:
    # Sum(Mod(n, 3), (n, 0, 10)) -> \sum_{n=0}^{10} n \bmod 3 (reads (sum n) mod 3); Product, Limit,
    # Derivative, Integral likewise
    body = Mod(n, 3)
    assert latex(Sum(body, (n, 0, 10))) == r'\sum_{n=0}^{10} \left(n \bmod 3\right)'
    assert latex(Product(body, (n, 0, 10))) == r'\prod_{n=0}^{10} \left(n \bmod 3\right)'
    assert latex(Limit(body, n, 0)) == r'\lim_{n \to 0^+} \left(n \bmod 3\right)'
    assert latex(Derivative(Mod(x, 3), x)) == r'\frac{d}{d x} \left(x \bmod 3\right)'
    assert latex(Integral(Mod(x, 3), x)) == r'\int \left(x \bmod 3\right)\, dx'


def test_adjoint_of_an_operator_with_a_body() -> None:
    # adjoint(Integral(A, x)) -> \int A\, dx^{\dagger}
    A = Symbol('A', commutative=False)
    assert latex(Adjoint(Integral(A, x))) == r'\left(\int A\, dx\right)^{\dagger}'
    assert latex(Adjoint(Derivative(f(x), x))) == r'\left(\frac{d}{d x} f{\left(x \right)}\right)^{\dagger}'


def test_float_in_scientific_notation_as_a_base() -> None:
    # Pow(Float('2.5e20'), 2) -> {2.5 \cdot 10^{20}}^{2}: braces only (sympy/sympy#21159)
    assert latex(Pow(Float('2.5e20'), 2)) == r'\left(2.5 \cdot 10^{20}\right)^{2}'
    assert latex(Pow(Float('2.5'), 2)) == r'2.5^{2}'


def test_algebraic_number_is_bracketed_as_its_expression() -> None:
    # x*AlgebraicNumber(sqrt(2), [1, 2]) -> x \sqrt{2} + 2; Pow(AlgebraicNumber, 2) -> \sqrt{2} + 2^{2}
    alpha = AlgebraicNumber(sqrt(2), [1, 2])
    assert latex(alpha) == r'\sqrt{2} + 2'
    assert latex(Mul(x, alpha)) == r'x \left(\sqrt{2} + 2\right)'
    assert latex(Pow(alpha, 2)) == r'\left(\sqrt{2} + 2\right)^{2}'


def test_negative_products_inside_products() -> None:
    # Mul(2, Mul(-y, z)) -> 2 - y z; Mul(x, Mul(-y, z)) -> x - y z
    assert latex(Mul(2, Mul(-1, y))) == r'2 \left(- y\right)'
    assert latex(Mul(x, Mul(Mul(-1, y), z))) == r'x \left(- y\right) z'
    assert latex(Mul(x, Mul(-2, y))) == r'x \left(- 2 y\right)'
    assert latex(Mul(x, Integer(-2))) == r'x \left(-2\right)'


def test_condition_set_brackets_its_condition() -> None:
    # ConditionSet(x, Or(x > 1, x < 0), Reals) -> x \in \mathbb{R} \wedge x > 1 \vee x < 0
    assert latex(ConditionSet(x, Or(x > 1, x < 0), Reals)) == \
        r'\left\{x\; \middle|\; x \in \mathbb{R} \wedge \left(x > 1 \vee x < 0\right) \right\}'


def test_divisor_sigma_subscript() -> None:
    # divisor_sigma(n, k + 1) -> \sigma_k + 1\left(n\right); divisor_sigma(n, 10) -> \sigma_10
    assert latex(DivisorSigma(n, k + 1)) == r'\sigma_{k + 1}\left(n\right)'
    assert latex(DivisorSigma(n, 10)) == r'\sigma_{10}\left(n\right)'
    assert latex(divisor_sigma(n, k)**2) == r'\sigma^{2}_{k}\left(n\right)'


def test_fold_func_brackets() -> None:
    # latex(f(x)) -> fx; f(2*x) -> f2 x; \operatorname{asinh}x; \cos {- x}
    assert latex(f(x), fold_func_brackets=True) == r'f{\left(x \right)}'
    assert latex(f(2*x), fold_func_brackets=True) == r'f{\left(2 x \right)}'
    assert latex(asinh(x), fold_func_brackets=True) == r'\operatorname{asinh} {x}'
    assert latex(Cos(Mul(-1, x)), fold_func_brackets=True) == r'\cos{\left(- x \right)}'
    assert latex(sin(x), fold_func_brackets=True) == r'\sin {x}'
    assert latex(sin(2*x), fold_func_brackets=True) == r'\sin {2 x}'
    assert latex(sin(x)**2, fold_func_brackets=True) == r'\sin^{2} {x}'
    assert latex(sin(x + y), fold_func_brackets=True) == r'\sin{\left(x + y \right)}'
    assert latex(sin(x*cos(y)), fold_func_brackets=True) == r'\sin{\left(x \cos {y} \right)}'


# ---------------------------------------------------------------------
# settings ignored, lower severity

def test_decimal_comma_switches_every_separator() -> None:
    # Interval(1.5, 2.5) -> \left[1{,}5, 2{,}5\right]; f(1.5, 2.5); Max(x, 1.5)
    half, more = Float(1.5), Float(2.5)
    assert latex(Interval(half, more), decimal_separator='comma') == r'\left[1{,}5; 2{,}5\right]'
    assert latex(f(half, more), decimal_separator='comma') == r'f{\left(1{,}5;2{,}5 \right)}'
    assert latex(Max(x, half), decimal_separator='comma') == r'\max\left(1{,}5; x\right)'
    assert latex([half, more], decimal_separator='comma') == r'\left[ 1{,}5; \  2{,}5\right]'


def test_fold_settings_on_a_rational_base() -> None:
    # Pow(1/2, -2/3) -> \frac{1}{(\frac{1}{2})^{2/3}}: plain brackets, a slash without fold_frac_powers
    e = Pow(Rational(1, 2), Rational(-2, 3))
    assert latex(e) == r'\frac{1}{\left(\frac{1}{2}\right)^{\frac{2}{3}}}'
    assert latex(e, fold_frac_powers=True) == r'\left(\frac{1}{2}\right)^{-2/3}'
    assert latex(e, fold_short_frac=True) == r'\frac{1}{\left(1 / 2\right)^{2 / 3}}'
    assert latex(Pow(Rational(1, 2), -1)) == r'\frac{1}{\frac{1}{2}}'


def test_built_in_names_are_not_split() -> None:
    # atan2(y, x) -> \operatorname{atan}_{2}; erf2; jacobi_symbol -> \operatorname{jacobi}_{symbol}
    assert latex(atan2(y, x)) == r'\operatorname{atan2}{\left(y,x \right)}'
    assert latex(erf2(x, y)) == r'\operatorname{erf2}{\left(x,y \right)}'
    assert latex(JacobiSymbol(n, k)) == r'\operatorname{jacobi\_symbol}{\left(n,k \right)}'


def test_domains_of_polynomials() -> None:
    # the monospace fallback of the domains
    assert latex(Poly(a*x**2 + 1, x)) == \
        r'\operatorname{Poly}{\left( a x^{2} + 1, x, domain=\mathbb{Z}\left[a\right] \right)}'


def test_information_is_not_lost() -> None:
    # exp_polar(x) -> e^{x}; unevaluated Mul(x, 3) -> x 3; Mul(sin(2), 3) -> \sin{\left(2 \right)} 3
    assert latex(ExpPolar(x)) == r'\operatorname{exp\_polar}{\left(x \right)}'
    assert latex(Mul(x, 3)) == r'x \cdot 3'
    assert latex(Mul(sin(2), 3)) == r'\sin{\left(2 \right)} \cdot 3'
