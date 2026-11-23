"""The LaTeX printer against SymPy's on a corpus where SymPy is right,
under the settings which do not touch the deliberate differences; the
deliberate differences with their expected output."""
from __future__ import annotations

import sympy

from sympy_extras.core import Integer, Rational, Symbol
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
