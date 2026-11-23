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
from typing import Optional, Sequence

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

__all__ = ['sstr', 'srepr', 'StrPrinter', 'ordered_terms', 'printed_terms', 'printed_factors']

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
        # 1/x and 1/sqrt(x) are printed as quotients: as a base, or in a
        # denominator, they need their parentheses (1/(1/x), (1/x)**2)
        return PREC_MUL if _printed_as_quotient(e) else PREC_POW
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


def _printed_as_quotient(e: Pow) -> bool:
    """Whether ``print_pow`` prints the power as ``1/...``."""
    exp = e.exp
    return isinstance(exp, Rational) and exp.p == -1 and exp.q in (1, 2) and e.base.is_commutative is not False


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
            # the sums, products and powers are printed here, the other
            # nodes print themselves through the methods of NodePrinter
            if isinstance(e, Add):
                return self.print_add(e.args)
            if isinstance(e, Mul):
                return self.print_mul(e.args)
            if isinstance(e, Pow):
                return self.print_pow(e.base, e.exp)
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
        parts: list[str] = []
        for i, term in enumerate(printed_terms(terms)):
            s = self.parenthesize(term, PREC_ADD)
            if i == 0:
                parts.append(s)
            elif s.startswith('-'):
                parts.append(' - ' + s[1:])
            else:
                parts.append(' + ' + s)
        return ''.join(parts)

    def print_mul(self, factors: Sequence[Expr]) -> str:
        factors = printed_factors(factors)
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
            # (SymPy's _print_Mul: only the commutative powers go to the denominator)
            if isinstance(f, Pow) and _negative_exponent(f.exp) and f.is_commutative is not False:
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
            # (SymPy's _print_Pow: the 1/... forms for commutative powers only)
            if exp.p == -1 and exp.q == 2 and base.is_commutative is not False:
                return '1/sqrt(%s)' % self.doprint(base)
            if exp.p == -1 and exp.q == 1 and base.is_commutative is not False:
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
    from .function import AppliedUndef, DefinedFunction, WildFunction
    from .numbers import Float
    from .symbol import Dummy, Symbol, Wild
    if isinstance(e, Wild):
        # (the exclusions are part of the wild; the predicates cannot be printed)
        if e.exclude:
            return "Wild(%r, exclude=[%s])" % (e.name, ', '.join(srepr(a) for a in e.exclude))
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
        # (with the assumptions of the function, as for a Symbol: f real is not f)
        given = ''.join(', %s=%r' % kv for kv in sorted(e.function.assumptions.items()))
        return "Function(%r%s)(%s)" % (e.name, given, ', '.join(srepr(a) for a in e.args))
    if isinstance(e, DefinedFunction):
        # the evaluating function (SymPy's name) for a node it returns, the
        # class for a node it would evaluate: sin(Symbol('x')), Sin(Integer(0))
        function_head = e.name if e.is_evaluated() else type(e).__name__
        return "%s(%s)" % (function_head, ', '.join(srepr(a) for a in e.args))
    head = e._srepr_name()
    if head is not None:
        return "%s(%s)" % (head, ', '.join(srepr(a) for a in e.args))
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
    SymPy's ``decompose_power``) and likewise ``exp(-2*z/3)`` is
    ``(exp(z/3), -2)``; ``(f, 1)`` when the exponent has no rational
    coefficient."""
    from .power import pow
    b, e = f.as_base_exp()
    if isinstance(e, Rational):
        if e.q == 1:
            return (b, e.p)
        return (pow(b, Rational(1, e.q)), e.p)
    c, rest = e.as_coeff_Mul()
    if isinstance(c, Rational) and not (c.p == 1 and c.q == 1):
        if c.q == 1:
            return (pow(b, rest), c.p)
        from .mul import mul
        return (pow(b, mul(Rational(1, c.q), rest)), c.p)
    if isinstance(c, Float) and c.as_fraction() < 0:
        # SymPy's as_coeff_Mul(rational=True) takes the sign of a float: exp(-2.5*z) is (exp(2.5*z), -1)
        from .mul import mul
        return (pow(b, mul(Float(number_from_fraction(-c.as_fraction()), precision=c.precision), rest)), -1)
    return (f, 1)


def _monomial_key(term: Expr, gens: list[Expr], values: NumericValues) -> tuple[Fraction, ...]:
    exponents: dict[Expr, Fraction] = {}
    _, rest = term.as_coeff_Mul()
    factors = rest.args if rest.is_Mul else (rest,)
    from .expr import UnevaluatedExpr
    for f in factors:
        if not isinstance(f, Expr) or isinstance(f, (ImaginaryUnit, NumberSymbol, UnevaluatedExpr)) or _is_numeric(f, values):
            continue
        b, n = _decompose_power(f)
        if isinstance(b, NumberSymbol):
            continue
        exponents[b] = exponents.get(b, Fraction(0)) + Fraction(n)
    return tuple(-exponents.get(g, Fraction(0)) for g in gens)


#: the values of the factors without free symbols met while the terms of
#: one sum are ordered (the value of an integral is computed once)
NumericValues = dict[Expr, Optional[complex]]


def _numeric_value(f: Expr, values: NumericValues) -> Optional[complex]:
    """The value of a factor without free symbols, ``None`` when it has
    none (``Sum(f(k), (k, 0, 1))``, an integral of an undefined function)."""
    if f in values:
        return values[f]
    from .evalf import to_complex
    value: Optional[complex] = None
    try:
        value = to_complex(f)
    except (TypeError, ValueError, OverflowError):
        pass
    values[f] = value
    return value


def _is_numeric(f: Expr, values: NumericValues) -> bool:
    """Whether the factor counts in the numerical coefficient of its
    term: SymPy's ``as_terms`` takes the factors without free symbols
    which have a numerical value; one which has none is a generator
    (``sqrt(2)*Sum(f(k), (k, 0, 1)) - 4/3``, the number last)."""
    return f.is_number and _numeric_value(f, values) is not None


def _generators(term: Expr, values: NumericValues) -> list[Expr]:
    _, rest = term.as_coeff_Mul()
    factors = rest.args if rest.is_Mul else (rest,)
    gens: list[Expr] = []
    from .expr import UnevaluatedExpr
    for f in factors:
        if not isinstance(f, Expr) or isinstance(f, (Number, ImaginaryUnit, NumberSymbol, UnevaluatedExpr)) or _is_numeric(f, values):
            continue
        if f.is_commutative is not True:
            # SymPy's as_terms: the factors which are not known to commute
            # (the noncommutative ones, and a Limit, whose commutativity is
            # unknown) are not generators
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
    from .accumbounds import AccumBounds
    # the coefficient: a number, or accumulation bounds (SymPy's flatten
    # puts them in the slot of the coefficient)
    rest = args[1:] if args and isinstance(args[0], (Number, AccumBounds)) else args
    return list(rest) == sorted(rest, key=compare_key)


def _printing_order(factors: Sequence[Expr]) -> list[Expr]:
    """The factors of an evaluated product (its commutative factors in the
    compare order) in the printing order (``default_sort_key``: the
    numbers, then ``sqrt(2)``, ``x``, ``sin(x)``, ``cos(x)``); the factors
    of an unevaluated product as given."""
    # (SymPy's args_cnc: the factors which are not known to commute, a
    # Limit among them, follow the commutative ones in their order)
    commutative = [f for f in factors if f.is_commutative is True]
    if not _in_compare_order(commutative):
        return list(factors)
    tail = [f for f in factors if f.is_commutative is not True]
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
    values: NumericValues = {}
    gens: set[Expr] = set()
    for t in terms:
        gens.update(_generators(t, values))
    ordered_gens = sorted(gens, key=default_sort_key)
    keys: dict[Expr, tuple[tuple[Fraction, ...], tuple[SortKey, ...], CoefficientRank, SortKey]] = {
        t: (_monomial_key(t, ordered_gens, values), _noncommutative_key(t), _coefficient_rank(t, values), t.sort_key())
        for t in terms}
    plain = sorted([t for t in terms if not t.is_Order], key=lambda t: keys[t])
    orders = sorted([t for t in terms if t.is_Order], key=lambda t: keys[t])
    if orders:
        plain.reverse()  # a series is printed in increasing order: x - x**3/6 + O(x**5)
    return plain + orders


def printed_terms(terms: Sequence[Expr], order: Optional[str] = None) -> list[Expr]:
    """The terms of a sum in the order in which the printers (``sstr``,
    the LaTeX and the pretty printer) write them: the terms of an
    unevaluated sum, and all terms under the order ``'none'``, as they
    are given; the terms of an evaluated sum (a number first, then the
    compare order) in the printing order of :func:`ordered_terms`
    (``None`` or ``'lex'``), by decreasing total degree (``'grlex'``) or
    by decreasing total degree with the last generators first
    (``'grevlex'``).

    Examples
    ========

    >>> from sympy_extras.core import Add, Integer, symbols
    >>> from sympy_extras.core.printing import printed_terms
    >>> x, y = symbols('x y')
    >>> printed_terms((x + y**2 + x*y + 1).args), printed_terms((x + y**2 + x*y + 1).args, 'grlex')
    ([x*y, x, y**2, 1], [x*y, y**2, x, 1])
    >>> printed_terms(Add(Integer(1), y, x).args), printed_terms((x + y + 1).args, 'none') == list((x + y + 1).args)
    ([1, y, x], True)
    """
    if order not in (None, 'lex', 'grlex', 'grevlex', 'none'):
        raise ValueError("the order of the terms must be None, 'lex', 'grlex', 'grevlex' or 'none', not %r" % (order,))
    if order == 'none' or not _in_compare_order(terms):
        return list(terms)
    if order is None or order == 'lex':
        return ordered_terms(terms)
    values: NumericValues = {}
    gens: set[Expr] = set()
    for t in terms:
        gens.update(_generators(t, values))
    ordered_gens = sorted(gens, key=default_sort_key)
    keyed = [(_monomial_key(t, ordered_gens, values), t) for t in ordered_terms(terms)]
    if order == 'grlex':
        keyed.sort(key=lambda kt: (sum(kt[0]), kt[0]))
    else:
        keyed.sort(key=lambda kt: (sum(kt[0]), tuple(-x for x in reversed(kt[0]))))
    return [t for _, t in keyed]


def printed_factors(factors: Sequence[Expr], order: Optional[str] = None) -> list[Expr]:
    """The factors of a product in the order in which the printers write
    them: the factors of an unevaluated product, and all factors under
    the order ``'none'``, as they are given; the commutative factors of
    an evaluated product by ``default_sort_key`` (the numbers, then
    ``sqrt(2)``, ``x``, ``sin(x)``, ``cos(x)``), the others after them.

    Examples
    ========

    >>> from sympy_extras.core import Integer, Mul, sqrt, symbols
    >>> from sympy_extras.core.printing import printed_factors
    >>> x, y = symbols('x y')
    >>> printed_factors((2*y*x*sqrt(Integer(2))).args), printed_factors(Mul(y, x, Integer(2)).args)
    ([2, sqrt(2), x, y], [y, x, 2])
    """
    if order == 'none':
        return list(factors)
    return _printing_order(factors)


#: (UnevaluatedExpr last, (has an imaginary part, its value), (real part, imaginary part))
CoefficientRank = tuple[int, tuple[bool, float], tuple[float, float]]


def _coefficient_rank(term: Expr, values: NumericValues) -> CoefficientRank:
    """Among the terms with the same monomial, SymPy's order: by the
    increasing numerical coefficient (``-3*pi*x - x``, ``x + sqrt(2)*x``,
    ``1 + sqrt(2) + E + pi``), the terms with an imaginary coefficient
    after the others (``x + I*x``), the ``UnevaluatedExpr`` terms last."""
    from .expr import UnevaluatedExpr
    if isinstance(term, UnevaluatedExpr) or any(isinstance(f, UnevaluatedExpr) for f in term.args):
        return (1, (False, 0.0), (0.0, 0.0))
    from .expr import factors_of
    value = complex(1.0)
    for f in factors_of(term):
        if f.is_number:
            # (the product of the values, as SymPy's as_terms: a factor without a value is a generator)
            found = _numeric_value(f, values)
            if found is not None:
                try:
                    value *= found
                except OverflowError:
                    return (0, (False, 0.0), (0.0, 0.0))
    return (0, (value.imag != 0, value.imag), (value.real, value.imag))


def _noncommutative_key(term: Expr) -> tuple[SortKey, ...]:
    """The keys of the factors which are not known to commute, in order
    (``A*B`` before ``B*A``; a term with such a factor after the ones
    without: ``1 + Limit(f(x), x, 0)``, as SymPy's ``as_terms`` has it)."""
    from .expr import factors_of
    return tuple(f.sort_key() for f in factors_of(term) if f.is_commutative is not True)


def _leads_a_difference(number: Expr, term: Expr) -> bool:
    """SymPy's special case of a sum of two terms printed as ``1 - x``: a
    positive finite number (``pi`` too) and a single factor with a
    negative numerical coefficient (``-x``, ``-2*x**2``, ``-2.5*x``, ``-sqrt(2)``, but
    not ``-pi*x`` or ``-x*y``)."""
    if not isinstance(number, (Number, NumberSymbol)) or number.is_positive is not True \
            or number.is_finite is not True:
        return False
    c, rest = term.as_coeff_Mul()
    if rest.is_Mul:
        return False
    if isinstance(c, Rational):
        return c.p < 0
    # a negative float times a single factor (SymPy: 6.25 - 2.5*x); a float alone is not a product
    return isinstance(c, Float) and c.as_fraction() < 0 and not rest.is_Number
