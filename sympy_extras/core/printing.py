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
from .numbers import (AlgebraicNumber, Float, ImaginaryUnit, NegativeInfinity, Number, NumberSymbol, Rational,
    number_from_fraction)
from .sorting import SortKey, compare_key, default_sort_key

# the classes printed here (none of these modules imports this one at module level)
from .add import Add  # noqa: E402
from .logic import And, Equivalent, Implies, Not, Or, Xor  # noqa: E402
from .mod import Mod  # noqa: E402
from .mul import Mul  # noqa: E402
from .power import Pow  # noqa: E402
from .relational import Relational  # noqa: E402

__all__ = ['sstr', 'srepr', 'StrPrinter', 'ordered_terms']

PREC_XOR = 10
PREC_IMPLIES = 10
PREC_OR = 20
PREC_AND = 30
PREC_RELATIONAL = 35
PREC_ADD = 40
PREC_MUL = 50
PREC_POW = 60
PREC_FUNC = 70
PREC_NOT = 100
PREC_ATOM = 1000
#: the arguments of ``|``, ``^`` and ``&`` are parenthesized below these
#: levels (SymPy's ``BitwiseOr``, ``BitwiseXor``, ``BitwiseAnd``)
PREC_OR_ARGS = 36
PREC_XOR_ARGS = 37
PREC_AND_ARGS = 38


def precedence(e: Basic) -> int:
    """The precedence of the printed form of ``e``."""
    if isinstance(e, Add):
        return PREC_ADD
    if isinstance(e, Mul):
        return PREC_ADD if e.could_extract_minus_sign() else PREC_MUL
    if isinstance(e, Pow):
        return PREC_POW
    if isinstance(e, Relational):
        return PREC_FUNC if e.rel_op in ('==', '!=') else PREC_RELATIONAL
    if isinstance(e, And):
        return PREC_AND
    if isinstance(e, Or):
        return PREC_OR
    if isinstance(e, Xor):
        return PREC_XOR
    if isinstance(e, (Implies, Equivalent)):
        return PREC_IMPLIES
    if isinstance(e, Not):
        return PREC_NOT
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
        # an evaluated sum (a number first, then the terms in the compare
        # order) is printed in the printing order; an unevaluated one as given
        canonical = _in_compare_order(terms)
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
        factors = _printing_order(factors)
        sign = ''
        numerator: list[str] = []
        denominator: list[str] = []
        numerator_exprs: list[Expr] = []
        # a Mod factor is parenthesized, as SymPy prints 2*(Mod(x, 3)), except in -Mod(x, 3)
        bare_negation = len(factors) == 2 and isinstance(factors[0], Rational) and factors[0] == -1
        for i, f in enumerate(factors):
            if isinstance(f, Mod) and not bare_negation:
                numerator.append('(%s)' % self.doprint(f))
                numerator_exprs.append(f)
                continue
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
            if isinstance(f, Pow) and _negative_exponent(f.exp):
                # x**(-2) and 4**(-n) go to the denominator: -n/4**n
                from .mul import neg
                minus = neg(f.exp)
                inverse = f.base if minus.is_Number and minus == 1 else Pow(f.base, minus)
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
        if isinstance(base, Mod):
            return '(%s)**%s' % (self.doprint(base), self.parenthesize(exp, PREC_POW, strict=True))
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
    >>> srepr(2*x + 1), srepr(-3*x/8)
    ("Add(Mul(Integer(2), Symbol('x')), Integer(1))", "Mul(Integer(-1), Rational(3, 8), Symbol('x'))")
    """
    from .function import AppliedSymPyFunction, AppliedUndef, WildFunction
    from .numbers import Float
    from .symbol import Dummy, Symbol, Wild
    if isinstance(e, Wild):
        return "Wild(%r)" % e.name
    if isinstance(e, WildFunction):
        return "WildFunction(%r)" % e.name
    if isinstance(e, Dummy):
        return "Dummy(%r)" % e.name
    if isinstance(e, AlgebraicNumber):
        return "AlgebraicNumber(%s, [%s])" % (srepr(e.root), ', '.join(srepr(c) for c in e.coeffs()))
    if isinstance(e, Symbol):
        assumptions = ''.join(', %s=%r' % kv for kv in e.given_assumptions.items())
        return "Symbol(%r%s)" % (e.name, assumptions)
    if isinstance(e, Rational):
        if e.q == 1:
            return 'Integer(%d)' % e.p
        return 'Rational(%d, %d)' % (e.p, e.q)
    if isinstance(e, Float):
        return "Float(%r, precision=%d)" % (e.repr_str(), e.precision)
    if isinstance(e, AppliedUndef):
        return "Function(%r)(%s)" % (e.name, ', '.join(srepr(a) for a in e.args))
    if isinstance(e, AppliedSymPyFunction):
        return "%s(%s)" % (e.name, ', '.join(srepr(a) for a in e.args))
    if not e.args:
        return str(e)
    args: Sequence[Basic] = e.args
    if isinstance(e, Add) and _in_compare_order(e.args):
        args = ordered_terms(e.args)
    elif isinstance(e, Mul) and _in_compare_order(e.args):
        args = e.as_ordered_factors()
    return '%s(%s)' % (type(e).__name__, ', '.join(srepr(a) for a in args))


def _decompose_power(f: Expr) -> tuple[Expr, int]:
    """``(b, n)`` with ``f == b**n`` and ``n`` an integer: ``x**(2*y)`` is
    ``(x**y, 2)``, ``exp(-x)`` is ``(exp(x), -1)``, ``x**(3/2)`` is
    ``(sqrt(x), 3)`` (the generator of a radical is its root, as in
    SymPy's ``as_terms``), ``(f, 1)`` when the exponent has no integer
    coefficient."""
    from .power import pow
    b, e = f.as_base_exp()
    if isinstance(e, Rational):
        if e.q == 1:
            return (b, e.p)
        return (pow(b, Rational(1, e.q)), e.p)
    c, rest = e.as_coeff_Mul()
    if isinstance(c, Rational) and c.q == 1 and c.p != 1:
        return (pow(b, rest), c.p)
    return (f, 1)


def _monomial_key(term: Expr, gens: list[Expr]) -> tuple[Fraction, ...]:
    exponents: dict[Expr, Fraction] = {}
    _, rest = term.as_coeff_Mul()
    factors = rest.args if rest.is_Mul else (rest,)
    from .expr import UnevaluatedExpr
    for f in factors:
        if not isinstance(f, Expr) or isinstance(f, (ImaginaryUnit, NumberSymbol, UnevaluatedExpr)) or f.is_number:
            continue
        b, n = _decompose_power(f)
        if isinstance(b, NumberSymbol):
            continue
        exponents[b] = exponents.get(b, Fraction(0)) + Fraction(n)
    return tuple(-exponents.get(g, Fraction(0)) for g in gens)


def _generators(term: Expr) -> list[Expr]:
    _, rest = term.as_coeff_Mul()
    factors = rest.args if rest.is_Mul else (rest,)
    gens: list[Expr] = []
    from .expr import UnevaluatedExpr
    for f in factors:
        if not isinstance(f, Expr) or isinstance(f, (Number, ImaginaryUnit, NumberSymbol, UnevaluatedExpr)) or f.is_number:
            continue
        b, _ = _decompose_power(f)
        if isinstance(b, NumberSymbol):
            continue
        gens.append(b)
    return gens


def _negative_exponent(e: Expr) -> bool:
    """Whether the exponent has a negative numerical coefficient (``-2``,
    ``-n``, ``-y/2``, ``-0.5``)."""
    c = e.as_coeff_Mul()[0]
    return (isinstance(c, Rational) and c.p < 0) or (isinstance(c, Float) and c.as_fraction() < 0)


def _in_compare_order(args: Sequence[Expr]) -> bool:
    """Whether the arguments are as an evaluating function stores them: a
    number first (``zoo`` ranks after the symbols in the compare order,
    but ``zoo + y`` keeps it first), then the compare order."""
    rest = args[1:] if args and isinstance(args[0], Number) else args
    return list(rest) == sorted(rest, key=compare_key)


def _printing_order(factors: Sequence[Expr]) -> list[Expr]:
    """The factors of an evaluated product (its commutative factors in the
    compare order) in the printing order (``default_sort_key``: the
    numbers, then ``sqrt(2)``, ``x``, ``sin(x)``, ``cos(x)``); the factors
    of an unevaluated product as given."""
    commutative = [f for f in factors if f.is_commutative is not False]
    if not _in_compare_order(commutative):
        return list(factors)
    tail = [f for f in factors if f.is_commutative is False]
    return sorted(commutative, key=default_sort_key) + tail


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
    if len(terms) == 2 and _leads_a_difference(terms[0], terms[1]):
        return list(terms)  # SymPy's special case: 1 - x, 2 - pi (but -x - 2, -pi*x + 1, -x*y + 1)
    gens: set[Expr] = set()
    for t in terms:
        gens.update(_generators(t))
    ordered_gens = sorted(gens, key=default_sort_key)
    keys: dict[Expr, tuple[tuple[Fraction, ...], tuple[SortKey, ...], CoefficientRank, SortKey]] = {
        t: (_monomial_key(t, ordered_gens), _noncommutative_key(t), _coefficient_rank(t), t.sort_key())
        for t in terms}
    plain = sorted([t for t in terms if not t.is_Order], key=lambda t: keys[t])
    orders = sorted([t for t in terms if t.is_Order], key=lambda t: keys[t])
    if orders:
        plain.reverse()  # a series is printed in increasing order: x - x**3/6 + O(x**5)
    return plain + orders


#: (UnevaluatedExpr last, (has an imaginary part, its value), (real part, imaginary part))
CoefficientRank = tuple[int, tuple[bool, float], tuple[float, float]]


def _coefficient_rank(term: Expr) -> CoefficientRank:
    """Among the terms with the same monomial, SymPy's order: by the
    increasing numerical coefficient (``-3*pi*x - x``, ``x + sqrt(2)*x``,
    ``1 + sqrt(2) + E + pi``), the terms with an imaginary coefficient
    after the others (``x + I*x``), the ``UnevaluatedExpr`` terms last."""
    from .evalf import to_complex
    from .expr import UnevaluatedExpr
    from .mul import mul
    if isinstance(term, UnevaluatedExpr) or any(isinstance(f, UnevaluatedExpr) for f in term.args):
        return (1, (False, 0.0), (0.0, 0.0))
    from .expr import factors_of
    coefficient = mul(*[f for f in factors_of(term) if f.is_number])
    try:
        value = to_complex(coefficient)
    except (TypeError, ValueError, OverflowError):
        return (0, (False, 0.0), (0.0, 0.0))
    return (0, (value.imag != 0, value.imag), (value.real, value.imag))


def _noncommutative_key(term: Expr) -> tuple[SortKey, ...]:
    """The keys of the noncommutative factors, in order (``A*B`` before ``B*A``)."""
    from .expr import factors_of
    return tuple(f.sort_key() for f in factors_of(term) if f.is_commutative is False)


def _leads_a_difference(number: Expr, term: Expr) -> bool:
    """SymPy's special case of a sum of two terms printed as ``1 - x``: a
    positive finite number (``pi`` too) and a single factor with a
    negative rational coefficient (``-x``, ``-2*x**2``, ``-sqrt(2)``, but
    not ``-pi*x`` or ``-x*y``)."""
    if not isinstance(number, (Number, NumberSymbol)) or number.is_positive is not True \
            or number.is_finite is not True:
        return False
    c, rest = term.as_coeff_Mul()
    return isinstance(c, Rational) and c.p < 0 and not rest.is_Mul
