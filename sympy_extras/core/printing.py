"""The string form of expressions, in SymPy's notation.

``sstr`` prints as SymPy's ``StrPrinter`` does: sums with their terms in
decreasing degree (``x**2 + 2*x + 1``), products with a leading
coefficient and the negative powers as a denominator (``-x*y/(2*z)``),
``sqrt`` for the exponent ``1/2``, ``1/x`` for the exponent ``-1``,
relations as ``x < 1`` and ``Eq(x, y)``. ``srepr`` prints the tree
(``Add(Mul(Integer(2), Symbol('x')), Integer(1))``).

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Printing*,
       https://docs.sympy.org/latest/modules/printing.html#module-sympy.printing.str
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from .basic import Basic
from .expr import Expr
from .numbers import Float, ImaginaryUnit, NegativeInfinity, Number, NumberSymbol, Rational, number_from_fraction
from .sorting import SortKey, default_sort_key

__all__ = ['sstr', 'srepr', 'StrPrinter', 'ordered_terms']

PREC_RELATIONAL = 35
PREC_ADD = 40
PREC_MUL = 50
PREC_POW = 60
PREC_FUNC = 70
PREC_ATOM = 1000


def precedence(e: Basic) -> int:
    """The precedence of the printed form of ``e``."""
    from .add import Add
    from .mul import Mul
    from .power import Pow
    from .relational import Relational
    if isinstance(e, Add):
        return PREC_ADD
    if isinstance(e, Mul):
        return PREC_ADD if sstr(e).startswith('-') else PREC_MUL
    if isinstance(e, Pow):
        return PREC_POW
    if isinstance(e, Relational):
        return PREC_RELATIONAL
    if isinstance(e, Rational):
        if e.p < 0:
            return PREC_ADD
        return PREC_MUL if e.q > 1 else PREC_ATOM
    if isinstance(e, Float):
        return PREC_ADD if e.as_fraction() < 0 else PREC_ATOM
    if isinstance(e, NegativeInfinity):
        return PREC_ADD
    return PREC_ATOM


class StrPrinter:
    """The printer of :func:`sstr`: ``doprint`` prints a node, which prints
    its arguments through the same printer; a float inside an expression
    is printed without its trailing zeros (``2.5*x``), as SymPy does, a
    float alone with all its digits (``2.50000000000000``)."""

    def __init__(self) -> None:
        self.level: int = 0

    def doprint(self, e: Basic) -> str:
        if isinstance(e, Float) and self.level > 0:
            return e.short_str()
        self.level += 1
        try:
            return e._sympystr(self)
        finally:
            self.level -= 1

    def parenthesize(self, e: Basic, level: int, strict: bool = False) -> str:
        """``e`` printed, in parentheses when its precedence is below
        ``level`` (or equal, when ``strict``)."""
        p = precedence(e)
        s = self.doprint(e)
        if p < level or (strict and p <= level):
            return '(%s)' % s
        return s

    def generic(self, e: Basic) -> str:
        """``Class(args)``."""
        if not e.args:
            return type(e).__name__
        return '%s(%s)' % (type(e).__name__, ', '.join(self.doprint(a) for a in e.args))

    def print_add(self, terms: Sequence[Expr]) -> str:
        canonical = list(terms) == sorted(terms, key=default_sort_key)
        parts: list[str] = []
        for i, term in enumerate(ordered_terms(terms) if canonical else terms):
            s = self.parenthesize(term, PREC_ADD)
            if i == 0:
                parts.append(s)
            elif s.startswith('-'):
                parts.append(' - ' + s[1:])
            else:
                parts.append(' + ' + s)
        return ''.join(parts)

    def print_mul(self, factors: Sequence[Expr]) -> str:
        from .power import Pow
        sign = ''
        numerator: list[str] = []
        denominator: list[str] = []
        numerator_exprs: list[Expr] = []
        for i, f in enumerate(factors):
            if isinstance(f, Rational) and i == 0:
                p, q = f.p, f.q
                if p < 0:
                    sign, p = '-', -p
                if p != 1 or (q == 1 and len(factors) == 1):
                    numerator.append(str(p))
                    numerator_exprs.append(Rational(p))
                if q != 1:
                    denominator.append(str(q))
                continue
            if isinstance(f, Float) and i == 0 and f.as_fraction() < 0:
                sign = '-'
                numerator.append(self.doprint(Float(number_from_fraction(-f.as_fraction()), precision=f.precision)))
                numerator_exprs.append(f)
                continue
            if isinstance(f, Pow) and isinstance(f.exp, Rational) and f.exp.p < 0:
                minus = Rational(-f.exp.p, f.exp.q)
                inverse = f.base if minus.p == 1 and minus.q == 1 else Pow(f.base, minus)
                denominator.append(self.parenthesize(inverse, PREC_MUL, strict=True))
                continue
            numerator.append(self.parenthesize(f, PREC_MUL, strict=i > 0 and isinstance(f, Number)))
            numerator_exprs.append(f)
        if not numerator:
            numerator.append('1')
        num = '*'.join(numerator)
        if not denominator:
            return sign + num
        den = denominator[0] if len(denominator) == 1 else '(%s)' % '*'.join(denominator)
        return '%s%s/%s' % (sign, num, den)

    def print_pow(self, base: Expr, exp: Expr) -> str:
        if isinstance(exp, Rational):
            if exp.p == 1 and exp.q == 2:
                return 'sqrt(%s)' % self.doprint(base)
            if exp.p == -1 and exp.q == 2:
                return '1/sqrt(%s)' % self.doprint(base)
            if exp.p == -1 and exp.q == 1:
                return '1/%s' % self.parenthesize(base, PREC_MUL, strict=True)
        b = self.parenthesize(base, PREC_POW, strict=True)
        e = self.parenthesize(exp, PREC_POW, strict=True)
        return '%s**%s' % (b, e)


def sstr(e: Basic) -> str:
    """The string form of ``e`` in SymPy's notation.

    Examples
    ========

    >>> from sympy_extras.core import symbols, sstr, Rational
    >>> x, y, z = symbols('x y z')
    >>> sstr(x**2 + 2*x + 1), sstr(-x*y/(2*z)), sstr(x**-2), sstr((x + 1)/(x + 2))
    ('x**2 + 2*x + 1', '-x*y/(2*z)', 'x**(-2)', '(x + 1)/(x + 2)')
    """
    return StrPrinter().doprint(e)


def srepr(e: Basic) -> str:
    """The tree of ``e`` as constructor calls.

    Examples
    ========

    >>> from sympy_extras.core import symbols, srepr
    >>> x = symbols('x')
    >>> srepr(2*x + 1)
    Add(Integer(1), Mul(Integer(2), Symbol('x')))
    """
    from .function import AppliedSymPyFunction, AppliedUndef
    from .numbers import Float
    from .symbol import Dummy, Symbol
    if isinstance(e, Dummy):
        return "Dummy(%r)" % e.name
    if isinstance(e, Symbol):
        assumptions = ''.join(', %s=%r' % kv for kv in e.given_assumptions.items())
        return "Symbol(%r%s)" % (e.name, assumptions)
    if isinstance(e, Rational):
        if e.q == 1:
            return 'Integer(%d)' % e.p
        return 'Rational(%d, %d)' % (e.p, e.q)
    if isinstance(e, Float):
        return "Float(%r, precision=%d)" % (str(e), e.precision)
    if isinstance(e, AppliedUndef):
        return "Function(%r)(%s)" % (e.name, ', '.join(srepr(a) for a in e.args))
    if isinstance(e, AppliedSymPyFunction):
        return "%s(%s)" % (e.name, ', '.join(srepr(a) for a in e.args))
    if not e.args:
        return str(e)
    return '%s(%s)' % (type(e).__name__, ', '.join(srepr(a) for a in e.args))


def _monomial_key(term: Expr, gens: list[Expr]) -> tuple[Fraction, ...]:
    exponents: dict[Expr, Fraction] = {}
    _, rest = term.as_coeff_Mul()
    factors = rest.args if rest.is_Mul else (rest,)
    for f in factors:
        if not isinstance(f, Expr) or isinstance(f, (ImaginaryUnit, NumberSymbol)):
            continue
        b, e = f.as_base_exp()
        if isinstance(b, NumberSymbol):
            continue
        if isinstance(e, Rational) and e.q == 1:
            exponents[b] = exponents.get(b, Fraction(0)) + Fraction(e.p)
        else:
            exponents[f] = exponents.get(f, Fraction(0)) + Fraction(1)
    return tuple(-exponents.get(g, Fraction(0)) for g in gens)


def _generators(term: Expr) -> list[Expr]:
    _, rest = term.as_coeff_Mul()
    factors = rest.args if rest.is_Mul else (rest,)
    gens: list[Expr] = []
    for f in factors:
        if not isinstance(f, Expr) or isinstance(f, (Number, ImaginaryUnit, NumberSymbol)):
            continue
        b, e = f.as_base_exp()
        if isinstance(b, NumberSymbol):
            continue
        gens.append(b if isinstance(e, Rational) and e.q == 1 else f)
    return gens


def ordered_terms(terms: Sequence[Expr]) -> list[Expr]:
    """The terms of a sum in the printing order: decreasing degrees in
    the generators (lexicographically), numbers last.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras.core.printing import ordered_terms
    >>> x, y = symbols('x y')
    >>> ordered_terms([y, x*y, x**2, x, Integer(1)])
    [x**2, x*y, x, y, 1]
    """
    if len(terms) == 2 and isinstance(terms[0], (Number, NumberSymbol)) and terms[1].could_extract_minus_sign():
        return list(terms)  # SymPy's special case: 1 - x
    gens: set[Expr] = set()
    for t in terms:
        gens.update(_generators(t))
    ordered_gens = sorted(gens, key=default_sort_key)
    keys: dict[Expr, tuple[tuple[Fraction, ...], int, SortKey]] = {
        t: (_monomial_key(t, ordered_gens), _constant_rank(t), t.sort_key()) for t in terms}
    return sorted(terms, key=lambda t: keys[t])


def _constant_rank(term: Expr) -> int:
    """Among the constant terms: numbers first, ``I`` last."""
    if isinstance(term, Number):
        return 0
    if term.has(ImaginaryUnit):
        return 2
    return 1
