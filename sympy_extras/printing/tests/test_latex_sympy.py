"""The LaTeX printer against SymPy's on a corpus where SymPy is right,
under the settings which do not touch the deliberate differences; the
deliberate differences with their expected output."""
from __future__ import annotations

import sympy

from sympy_extras.core import Integer, Rational, Symbol, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.interop import to_sympy
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys.polytools import Poly
from sympy_extras.printing import latex
from sympy_extras.printing.latex import LatexPrinter, LatexSettings
from sympy_extras.printing.tests.corpus import DIFFERENT, SAME

x, y = Symbol('x'), Symbol('y')

#: settings under which the whole corpus prints as in SymPy
SETTINGS: list[LatexSettings] = [
    LatexSettings(),
    LatexSettings(fold_frac_powers=True),
    LatexSettings(mul_symbol='dot'),
    LatexSettings(mul_symbol='times', order='none'),
    LatexSettings(inv_trig_style='full'),
    LatexSettings(ln_notation=True, gothic_re_im=True, imaginary_unit='rj'),
    LatexSettings(root_notation=False),
    LatexSettings(long_frac_ratio=2),
    LatexSettings(itex=True, mode='equation'),
    LatexSettings(mode='equation*'),
    LatexSettings(parenthesize_super=False, full_prec=True),
    LatexSettings(min=-2, max=3),
    LatexSettings(order='grlex'),
    LatexSettings(order='grevlex'),
    # the other orders of SymPy's printers (they raised ValueError)
    LatexSettings(order='rev-lex'),
    LatexSettings(order='rev-grlex'),
    LatexSettings(order='rev-grevlex'),
    LatexSettings(order='igrlex'),
    LatexSettings(order='rev-ilex'),
    # (with a product sign, as for 'none': the factors are printed as stored under both, a number may then follow a
    # symbol, and SymPy writes y 2^{- x} where this printer puts a dot)
    LatexSettings(mul_symbol='times', order='old'),
]


def _sympy_latex(e: Basic, s: LatexSettings) -> str:
    return str(sympy.latex(to_sympy(e), mode=s.mode, fold_frac_powers=s.fold_frac_powers, itex=s.itex,
                           inv_trig_style=s.inv_trig_style, ln_notation=s.ln_notation,
                           long_frac_ratio=s.long_frac_ratio, mul_symbol=s.mul_symbol, order=s.order,
                           root_notation=s.root_notation, imaginary_unit=s.imaginary_unit,
                           gothic_re_im=s.gothic_re_im, parenthesize_super=s.parenthesize_super, min=s.min,
                           max=s.max, full_prec=s.full_prec))


def test_corpus_prints_as_in_sympy() -> None:
    assert len(SAME) > 350
    for s in SETTINGS:
        printer = LatexPrinter(s)
        failures = [(str(e), printer.doprint(e), _sympy_latex(e, s)) for e in SAME
                    if printer.doprint(e) != _sympy_latex(e, s)]
        assert not failures, (s, failures[:5], len(failures))


def test_deliberate_differences() -> None:
    for e, expected, reason in DIFFERENT:
        assert latex(e) == expected, (str(e), reason)


def test_matrices_as_in_sympy() -> None:
    m = ExprMatrix.from_rows([[x, y], [1, x**2/2]])
    reference = sympy.Matrix(2, 2, [to_sympy(v) for v in m.entries])
    assert latex(m) == sympy.latex(reference)
    assert latex(m, mat_delim='(', mat_str='array') == sympy.latex(reference, mat_delim='(', mat_str='array')
    assert latex(m, mode='inline') == sympy.latex(reference, mode='inline')
    assert latex(m, mat_delim='') == sympy.latex(reference, mat_delim='')


def test_poly() -> None:
    # as SymPy, with a space between the powers of a monomial and after a leading minus sign
    assert latex(Poly(x**2 + 2*x + 1, x)) == sympy.latex(sympy.Poly(to_sympy(x**2 + 2*x + 1), to_sympy(x)))
    assert latex(Poly(x**2*y - y + 3, x, y)) == \
        r'\operatorname{Poly}{\left( x^{2} y - y + 3, x, y, domain=\mathbb{Z} \right)}'
    assert latex(Poly(-x**3 + x/2, x)) == \
        r'\operatorname{Poly}{\left( - x^{3} + \frac{1}{2} x, x, domain=\mathbb{Q} \right)}'


def test_python_values_as_in_sympy() -> None:
    sx = to_sympy(x)
    assert latex([x, Integer(1)]) == sympy.latex([sx, 1])
    assert latex((x, Rational(1, 2))) == sympy.latex((sx, sympy.Rational(1, 2)))
    assert latex((x,)) == sympy.latex((sx,))
    assert latex(3) == sympy.latex(3) and latex(-3) == sympy.latex(-3) and latex(2.5) == sympy.latex(2.5)
    assert latex(True) == sympy.latex(True) and latex(None) == sympy.latex(None)
    assert latex({x, Integer(2)}) == sympy.latex({sx, 2})


def test_wrapped_objects_of_sympy_print_as_in_sympy() -> None:
    # a wrapped object of SymPy (interop's SymPyExprNode, ...) printed as the name of its wrapper,
    # \operatorname{SymPyExprNode}\left(\text{SymPyNode}, 2, 2\right) for MatrixSymbol('A', 2, 2): it is now printed by
    # SymPy's own LaTeX printer, and bracketed by ours from the precedence SymPy gives it
    from sympy_extras.core.expr import as_expr
    from sympy_extras.interop import from_sympy
    sx = sympy.Symbol('x')
    A = sympy.MatrixSymbol('A', 2, 2)
    wrapped = [A, A.T, A*A, sympy.Trace(A), sympy.Determinant(A), sympy.Identity(3), sympy.Indexed('a', 1),
               sympy.Integral(sympy.Function('f')(sx)/2, (sx, 0, 1)), sympy.KroneckerProduct(A, A),
               sympy.Range(0, 5), sympy.ImmutableMatrix([[1, sx]])]
    for w in wrapped:
        node = from_sympy(w)
        assert type(node).__name__.startswith('SymPy'), w
        assert latex(node) == sympy.latex(w), w
        assert 'SymPy' not in latex(node)
    assert latex(from_sympy(A)) == 'A' and latex(from_sympy(sympy.Trace(A))) == r'\operatorname{tr}\left(A \right)'
    # inside native nodes: the brackets are ours, decided from how the foreign output binds
    trace = as_expr(from_sympy(sympy.Trace(A)))
    integral = as_expr(from_sympy(sympy.Integral(sx, sx)))
    for e in (trace + x, x*trace, trace**2, 2**trace, trace/y, -trace, x - trace, integral + x, x*integral, integral**2,
              integral/y, sqrt(integral)):
        assert latex(e) == sympy.latex(to_sympy(e, evaluate=False)), str(e)
    assert latex(integral**2) == r'\left(\int x\, dx\right)^{2}'
    # the settings SymPy's printer shares are handed over
    half = as_expr(from_sympy(sympy.Integral(sympy.Function('f')(sx)/2, (sx, 0, 1))))
    assert latex(half + x, mode='inline') == r'$x + \int\limits_{0}^{1} f{\left(x \right)} / 2\, dx$'
    assert latex(as_expr(from_sympy(sympy.Integral(2*sx*sympy.Symbol('y'), sx))), mul_symbol='dot') == \
        sympy.latex(sympy.Integral(2*sx*sympy.Symbol('y'), sx), mul_symbol='dot')
    # an atom of SymPy (the name of the matrix symbol) and a container with wrapped objects
    assert latex(from_sympy(A).args[0]) == 'A'
    assert latex(from_sympy(sympy.Tuple(A, A.T))) == sympy.latex(sympy.Tuple(A, A.T))


def test_applied_functions_of_sympy_print_in_the_notation_of_sympy() -> None:
    # a function of SymPy applied to native arguments (interop's AppliedSymPyFunction) printed the name of its class,
    # \operatorname{elliptic\_k}{\left(x \right)} and \operatorname{Ynm}{\left(1,0,x,y \right)}: it is printed by
    # SymPy's own LaTeX printer, as the wrapped objects are, the node as it stands (not evaluated)
    from sympy_extras.interop import AppliedSymPyFunction
    sx, sy = sympy.Symbol('x'), sympy.Symbol('y')
    assert latex(AppliedSymPyFunction(sympy.elliptic_k, x)) == r'K\left(x\right)'
    assert latex(AppliedSymPyFunction(sympy.Ynm, 1, 0, x, y)) == sympy.latex(sympy.Ynm(1, 0, sx, sy))
    assert latex(AppliedSymPyFunction(sympy.mathieuc, x, y, 1)) == sympy.latex(sympy.mathieuc(sx, sy, 1))
    assert latex(AppliedSymPyFunction(sympy.exp, x)) == 'e^{x}'
    gamma = AppliedSymPyFunction(sympy.gamma, x + y)
    assert latex(gamma) == sympy.latex(sympy.gamma(sx + sy)) and 'operatorname' not in latex(gamma)
    # an operand: the brackets are ours, and the printed form reads as the expression
    assert latex(gamma + x) == sympy.latex(sympy.gamma(sx + sy) + sx)
    assert latex(gamma**2) == r'\left(\Gamma\left(x + y\right)\right)^{2}'
    assert latex(AppliedSymPyFunction(sympy.exp, x)**2) == r'\left(e^{x}\right)^{2}'
    # not evaluated by the printing: gamma(1) stays
    assert latex(AppliedSymPyFunction(sympy.gamma, 1)) == r'\Gamma\left(1\right)'


def test_derivative_of_symbolic_order() -> None:
    # a derivative of symbolic order (Derivative(f(x), (x, n)), which the node could not hold) prints its orders
    # as exponents, and their sum on the operator, as SymPy's printer does
    from sympy_extras.core import Derivative, Function
    n, m = Symbol('n', integer=True), Symbol('m', integer=True)
    f = Function('f')
    assert latex(Derivative(f(x), (x, n))) == r'\frac{d^{n}}{d x^{n}} f{\left(x \right)}'
    for d in (Derivative(f(x), (x, n)), Derivative(f(x, y), (x, n), y, (x, m + 1)), Derivative(x**3, (x, 2*n)),
              Derivative(f(x, y), (x, n), (y, 2)), Derivative(f(x), (x, n))**2 + 1):
        assert latex(d) == sympy.latex(to_sympy(d, evaluate=False)), str(d)
