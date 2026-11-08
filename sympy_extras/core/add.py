"""Sums: the class ``Add`` and the evaluating function ``add``.

``Add(*terms)`` is a node with the given terms, in the given order, which
is not simplified: ``Add(x, x)`` has the arguments ``(x, x)``. The function
``add(*terms)`` returns the canonical sum of SymPy: nested sums are
flattened, the numbers are summed, the terms with the same non-numerical
part are collected (``add(x, x)`` is ``2*x``), the zero terms are dropped,
the arguments are sorted (``add(y, x) == add(x, y)``) with the number
first, and a sum of one term is that term. The operators of ``Expr`` call
``add``; ``Add(...).doit()`` is ``add(...)``.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Add*,
       https://docs.sympy.org/latest/modules/core.html#sympy.core.add.Add

The common radicals of ``as_content_primitive(radical=True)`` are ported
from SymPy's ``sympy.core.add``.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/add.py (Add.as_content_primitive, the radicals) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the rest of the module is original code.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from .basic import Basic
from .containers import Tuple
from .expr import Expr, ExprLike, factors_of, to_expr
from .numbers import (ComplexInfinity, Infinity, Integer, NaN, NegativeInfinity, Number, Rational,
    nan, number_from_fraction)
from .operations import AssocOp
from .parameters import distribute, global_parameters
from .sorting import ArgsKey, ClassKey, SortKey, compare_key

if TYPE_CHECKING:
    from .order import Order
    from .printing import StrPrinter
    from .symbol import Symbol

__all__ = ['Add', 'add', 'sub', 'to_expr']


class Add(AssocOp):
    """An unevaluated sum.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Add
    >>> x, y = symbols('x y')
    >>> s = Add(x, 1, x)
    >>> s
    x + 1 + x
    >>> s.args
    (x, 1, x)
    >>> s.doit()
    2*x + 1
    """

    __slots__ = ()

    is_Add = True
    identity = Integer(0)

    @classmethod
    def class_key(cls) -> ClassKey:
        return (3, 1, 'Add')

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import add_facts
        return add_facts(self._args)

    # PORTED FROM SYMPY: sympy/core/add.py:Add.removeO
    def removeO(self) -> Expr:
        return _raw([t for t in self._args if not t.is_Order])

    # PORTED FROM SYMPY: sympy/core/add.py:Add.getO
    def getO(self) -> Optional[Expr]:
        orders = [t for t in self._args if t.is_Order]
        return _raw(orders) if orders else None

    # PORTED FROM SYMPY: sympy/core/add.py:Add._eval_nseries
    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        return add(*[t.nseries(x, n=n, logx=logx, cdir=cdir) for t in self._args])

    # PORTED FROM SYMPY: sympy/core/add.py:Add.extract_leading_order
    def extract_leading_order(self, symbols: Sequence[Symbol], point: Optional[Sequence[Expr]] = None
                              ) -> tuple[tuple[Expr, Order], ...]:
        """The terms of the leading order with their order terms.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y = symbols('x y')
        >>> (x + 1 + 1/x**5).extract_leading_order([x])
        ((x**(-5), O(x**(-5))),)
        >>> (1 + x).extract_leading_order([x]), (x + x**2).extract_leading_order([x])
        (((1, O(1)),), ((x, O(x)),))
        """
        from .order import Order, order
        pts: Sequence[Expr] = point if point else [Integer(0)] * len(symbols)
        lst: list[tuple[Expr, Order]] = []
        for ef in self._args:
            found = order(ef, *[Tuple(s, p) for s, p in zip(symbols, pts)])
            if not isinstance(found, Order):
                raise TypeError("the order of %s is %s" % (ef, found))
            of: Optional[Order] = found
            for _, o in lst:
                if o.contains(found) and o != found:
                    of = None
                    break
            if of is None:
                continue
            new_lst: list[tuple[Expr, Order]] = [(ef, of)]
            for e, o in lst:
                if of.contains(o) and o != of:
                    continue
                new_lst.append((e, o))
            lst = new_lst
        return tuple(lst)

    # PORTED FROM SYMPY: sympy/core/add.py:Add._eval_as_leading_term
    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        from .function import _expand_logs, expand_mul
        from .order import Order, order
        from .symbol import Dummy, Symbol as _Symbol
        from sympy_extras.functions.elementary.exponential import Log, log
        from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise_fold
        o = self.getO()
        if o is None:
            o = order(Integer(0))
        old = self.removeO()
        if old.has(Piecewise):
            old = piecewise_fold(old)
        # the last part of expand_log (which also calls expand_mul with
        # factor=True, more expensive)
        if any(isinstance(a, Log) for a in self._args):
            old = _expand_logs(old, True, False, False)
        expr = expand_mul(old)
        if not isinstance(expr, Add):
            return expr.as_leading_term(x, logx=logx, cdir=cdir)
        infinite = [t for t in expr._args if t.is_infinite is True]
        _logx: Expr = Dummy('logx') if logx is None else logx
        leading_terms = [t.as_leading_term(x, logx=_logx, cdir=cdir) for t in expr._args]
        smallest = order(Integer(0))
        new_expr: Expr = Integer(0)
        try:
            for term in leading_terms:
                o_term = order(term, x)
                if not _order_contains(smallest, o_term):
                    smallest = o_term
                    new_expr = term
                elif _order_contains(o_term, smallest):
                    new_expr = add(new_expr, term)
        except TypeError:
            return expr
        if logx is None:
            new_expr = new_expr.subs(_logx, log(x))
        is_zero = new_expr.is_zero
        if is_zero is None:
            new_expr = new_expr.trigsimp().cancel()
            is_zero = new_expr.is_zero
        if is_zero is True:
            # the simple analysis cancelled the terms, but a term must be
            # returned: the leading term of the series
            n0: Expr
            try:
                found_n = smallest.getn()
                n0 = Integer(1) if found_n is None else found_n
            except NotImplementedError:
                n0 = Integer(1)
            if n0.has(_Symbol):
                n0 = Integer(1)
            res: Expr = order(Integer(1))
            incr = 1
            while res.is_Order:
                res = old._eval_nseries(x, _ceiling_int(add(n0, Integer(incr))), logx, cdir)
                res = res.cancel().powsimp().trigsimp()
                incr *= 2
            return res.as_leading_term(x, logx=logx, cdir=cdir)
        if isinstance(new_expr, NaN):
            return add(_raw(infinite), o)
        return new_expr

    def as_content_primitive(self, radical: bool = False, clear: bool = True) -> tuple[Number, Expr]:
        # the content of each term comes out first ((1/4)**n becomes 4**(-n)), as SymPy does
        rebuilt = [_mul(*t.as_content_primitive(radical=radical, clear=clear)) for t in self._args]
        if any(r is not t for r, t in zip(rebuilt, self._args)) and rebuilt != list(self._args):
            total = add(*rebuilt)
            if isinstance(total, Add) and total != self:
                return total.as_content_primitive(radical=radical, clear=clear)
        c, p = self.primitive()
        if clear or not isinstance(c, Rational) or c.q == 1 or self._clearable():
            return (c, _common_radicals(p) if radical else p)
        # SymPy's clear=False: the denominator stays (x/2 + y); the integer content is taken
        if c.p == 1:
            return (Integer(1), _common_radicals(self) if radical else self)
        from .mul import mul
        rest = add(*[mul(Rational(1, c.p), t) for t in self._args])
        return (Integer(c.p), _common_radicals(rest) if radical else rest)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return add(*[_as_expr(a) for a in args])

    def _sympystr(self, printer: StrPrinter) -> str:
        return printer.print_add(self._args)

    def _subs_partial(self, old: Basic, new: Basic) -> Optional[Basic]:
        if not (isinstance(old, Add) and isinstance(new, Expr)):
            return None
        remaining = list(self._args)
        for term in old.args:
            if term not in remaining:
                return None
            remaining.remove(term)
        return add(new, *[_as_expr(t._subs(old, new)) for t in remaining])

    def sort_key(self) -> SortKey:
        # the arguments in the printing order, as SymPy's key does
        from .printing import ordered_terms
        key = Expr.sort_key(self)
        args: ArgsKey = (len(self._args), tuple(a.sort_key() for a in ordered_terms(self._args)))
        return (key[0], args, key[2], key[3])

    def as_coeff_Add(self) -> tuple[Number, Expr]:
        number: Number = Integer(0)
        rest: list[Expr] = []
        for a in self._args:
            if isinstance(a, Number):
                number = number.number_add(a)
            else:
                rest.append(a)
        return (number, _raw(rest))

    def as_two_terms(self) -> tuple[Expr, Expr]:
        return (self._args[0], _raw(list(self._args[1:])))

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        # the terms with the same denominator are summed first, as SymPy
        # does: 1 + (x + 1)/(x - 1) gives (2*x, x - 1)
        mul = _mul
        content, primitive = self.primitive()
        if not isinstance(primitive, Add):
            return mul(content, primitive).as_numer_denom()
        groups: dict[Expr, list[Expr]] = {}
        for a in primitive.args:
            n, d = a.as_numer_denom()
            groups.setdefault(d, []).append(n)
        cn, cd = content.as_numer_denom()
        if len(groups) == 1:
            # one denominator: the content multiplies each numerator (11 - 11*z, 6)
            d, ns = next(iter(groups.items()))
            with distribute(False):
                return (add(*[mul(cn, n) for n in ns]), mul(cd, d))
        denominators = list(groups)
        numerators = [mul(add(*ns), *[e for e in denominators if e is not d]) for d, ns in groups.items()]
        with distribute(False):  # SymPy: (2*(E*x*y + 1), 2*(E + 3))
            return (mul(cn, add(*numerators)), mul(cd, *denominators))

    def could_extract_minus_sign(self) -> bool:
        # SymPy: more negative terms than positive ones, or as many and the
        # sum sorting before its negation
        negatives = sum(1 for a in self._args if a.could_extract_minus_sign())
        positives = len(self._args) - negatives
        if negatives != positives:
            return negatives > positives
        return self.sort_key() < _mul(Integer(-1), self).sort_key()

    def _clearable(self) -> bool:
        """SymPy's rule for ``clear=False``: a fractional content is taken
        out only when every term has a fractional coefficient (``x/2 + y/4``
        gives ``(2*x + y)/4``, ``x/2 + y`` and ``x + 2/3`` stay)."""
        return all(_as_rational(a.as_coeff_Mul()[0]).q != 1 for a in self._args)

    def primitive(self) -> tuple[Number, Expr]:
        from .mul import mul
        coefficients = [a.as_coeff_Mul()[0] for a in self._args]
        values = [c.exact_value() for c in coefficients]
        if any(v is None for v in values) or any(c.is_Float for c in coefficients):
            return (Integer(1), self)
        g = Fraction(0)
        for v in values:
            if v is not None:
                g = _gcd_fractions(g, abs(v))
        if g == 0 or g == 1:
            return (Integer(1), self)
        content = number_from_fraction(g)
        inverse = number_from_fraction(1 / g)
        return (content, add(*[mul(inverse, a) for a in self._args]))


def _gcd_fractions(a: Fraction, b: Fraction) -> Fraction:
    from math import gcd, lcm
    return Fraction(gcd(a.numerator, b.numerator), lcm(a.denominator, b.denominator))


def _raw(terms: list[Expr]) -> Expr:
    """``Add(*terms)`` without evaluation, one term as itself, none as 0."""
    if not terms:
        return Integer(0)
    if len(terms) == 1:
        return terms[0]
    return Add(*terms)


def _as_expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


def _as_rational(n: Number) -> Rational:
    return n if isinstance(n, Rational) else Integer(1)


def add(*args: ExprLike) -> Expr:
    """The evaluated sum of the arguments.

    Examples
    ========

    >>> from sympy_extras.core import symbols, add, Rational
    >>> x, y = symbols('x y')
    >>> add(x, x)
    2*x
    >>> add(x, 2*y, 1, x, -1)
    2*x + 2*y
    >>> add(Rational(1, 2), Rational(1, 2))
    1
    >>> add(x, -x)
    0
    >>> add()
    0
    """
    mul = _mul
    stack = [to_expr(a) for a in args]
    if not global_parameters.evaluate:
        return _raw([t for a in stack for t in (a.args if isinstance(a, Add) else (a,))])
    if any(t.is_Order for a in stack for t in (a.args if isinstance(a, Add) else (a,))):
        return _with_orders(stack)
    # SymPy's AssocOp.__new__: the identities dropped, a single argument
    # returned as it is
    nonidentity = [t for t in stack if not (isinstance(t, Integer) and t.p == 0)]
    if len(nonidentity) == 1:
        return nonidentity[0]
    number: Number = Integer(0)
    coefficients: dict[Expr, Number] = {}
    for a in stack:
        if isinstance(a, Add):
            stack.extend(a.args)
            continue
        if isinstance(a, Number):
            number = number.number_add(a)
            continue
        c, rest = a.as_coeff_Mul()
        if rest in coefficients:
            coefficients[rest] = coefficients[rest].number_add(c)
        else:
            coefficients[rest] = c
    if isinstance(number, NaN):
        return nan
    terms: list[Expr] = []
    needs_pass = False
    for rest, c in coefficients.items():
        if isinstance(c, NaN):
            return nan
        value = c.exact_value()
        if value is not None and value == 0:
            continue
        if _absorbed_in_sum(number, rest, c):
            continue
        # PORTED FROM SYMPY: sympy/core/add.py:Add.flatten (the terms re-created)
        if isinstance(c, Rational) and c.p == 1 and c.q == 1:
            term = rest
            terms.append(term)
            needs_pass = needs_pass or term.is_Add or isinstance(term, Number)
            continue
        from .mul import Mul
        if isinstance(rest, Mul):
            # c in slot 0 of the product's arguments: a nested product
            # (x*y*(2*y + 1))/2 is flattened one level
            term = Mul(c, *rest.args)
        elif isinstance(rest, Add):
            term = Mul(c, rest)
        else:
            term = mul(c, rest)
        if term.is_Add or isinstance(term, Number):
            needs_pass = True
        terms.append(term)
    if needs_pass:
        return add(number, *terms)
    terms.sort(key=compare_key)
    nonzero = not (isinstance(number, Rational) and number.p == 0)
    if nonzero:
        terms.insert(0, number)
    if not terms:
        return number
    if len(terms) == 1:
        return terms[0]
    return Add(*terms)


# PORTED FROM SYMPY: sympy/core/add.py:Add.flatten (the order terms)
def _with_orders(stack: list[Expr]) -> Expr:
    """A sum with ``Order`` terms: the order terms contained in others
    dropped, then the terms contained in an order term (``x + O(x)`` is
    ``O(x)``, ``1 + O(1)`` is ``O(1)``)."""
    from .order import Order
    flat: list[Expr] = []
    pending = list(stack)
    while pending:
        a = pending.pop(0)
        if isinstance(a, Add):
            pending[:0] = list(a._args)
        else:
            flat.append(a)
    order_factors: list[Order] = []
    others: list[Expr] = []
    for o in flat:
        if isinstance(o, Order):
            if o.expr.is_zero is True:
                continue
            if any(o1.contains(o) for o1 in order_factors):
                continue
            order_factors = [o] + [o1 for o1 in order_factors if not o.contains(o1)]
        else:
            others.append(o)
    rest = add(*others)
    if isinstance(rest, NaN):
        return rest
    coeff: Expr = Integer(0)
    newseq: list[Expr] = []
    for t in (rest._args if isinstance(rest, Add) else (rest,)):
        if isinstance(t, Number):
            coeff = t
        else:
            newseq.append(t)
    newseq2: list[Expr] = [t for t in newseq if not any(o.contains(t) for o in order_factors)]
    terms: list[Expr] = newseq2 + list(order_factors)
    for o in order_factors:
        if o.contains(coeff):
            coeff = Integer(0)
            break
    terms.sort(key=compare_key)
    if not (isinstance(coeff, Integer) and coeff.p == 0):
        terms.insert(0, coeff)
    return _raw(terms)


def _absorbed_in_sum(number: Number, rest: Expr, c: Number) -> bool:
    """Whether the term ``c*rest`` is absorbed by the infinite ``number``:
    ``oo`` absorbs the real terms (``oo + p`` is ``oo``), ``zoo`` the finite
    terms which are known real or known non-real (``zoo + 3*I`` is ``zoo``),
    as SymPy does."""
    if isinstance(number, (Infinity, NegativeInfinity)):
        if not c.is_finite_number:
            return False
        return rest.is_real is True
    if isinstance(number, ComplexInfinity):
        if not c.is_finite_number:
            return False
        return rest.is_finite is True and rest.is_extended_real is not None
    return False


def sub(a: ExprLike, b: ExprLike) -> Expr:
    """The evaluated difference ``a - b``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, sub
    >>> x, y = symbols('x y')
    >>> sub(x, y), sub(x, x), sub(2*x, x)
    (x - y, 0, x)
    """
    from .mul import mul
    return add(a, mul(Integer(-1), b))

_MUL: Optional[Callable[..., Expr]] = None


# PORTED FROM SYMPY: sympy/core/add.py:Add.as_content_primitive (the radical branch)
def _common_radicals(prim: Expr) -> Expr:
    """``prim`` with the radicals of integers common to all its terms taken
    out: for each root index ``q`` present in every term, the gcd ``g`` of
    the products of the ``q``-th roots, ``g**(1/q)`` (``sqrt(2)*x +
    sqrt(6)`` is ``sqrt(2)*(x + sqrt(3))``)."""
    if not isinstance(prim, Add):
        return prim
    from math import gcd
    from .power import Pow, pow
    radicals: list[dict[int, int]] = []
    common: Optional[set[int]] = None
    for term in prim._args:
        products: dict[int, int] = {}
        for f in factors_of(term):
            if isinstance(f, Pow):
                b, e = f.as_base_exp()
                # a rational power of an integer is normalised with 0 < e < 1
                if isinstance(e, Rational) and isinstance(b, Integer) and e.p > 0:
                    products[e.q] = products.get(e.q, 1) * abs(b.p) ** e.p
        if not products:
            return prim
        common = set(products) if common is None else common & set(products)
        if not common:
            return prim
        radicals.append(products)
    if common is None:
        return prim
    taken: list[Expr] = []
    for q in sorted(common):
        g = 0
        for r in radicals:
            g = gcd(g, r[q])
        if g != 1:
            taken.append(pow(Integer(g), Rational(1, q)))
    if not taken:
        return prim
    G = _mul(*taken)
    inverse = pow(G, Integer(-1))
    return _mul(G, add(*[_mul(t, inverse) for t in prim._args]))



def _mul(*args: ExprLike) -> Expr:
    """``mul`` of the sibling module, bound at the first call (it imports this one)."""
    global _MUL
    if _MUL is None:
        from .mul import mul
        _MUL = mul
    return _MUL(*args)


def _order_contains(o: Expr, e: Expr) -> bool:
    """``e in o`` of SymPy: ``TypeError`` when ``contains`` cannot decide."""
    from .order import Order
    if not isinstance(o, Order):
        raise TypeError("an order term is expected, got %s" % (o,))
    result = o.contains(e)
    if result is None:
        raise TypeError('contains did not evaluate to a bool')
    return result


def _ceiling_int(e: Expr) -> int:
    """The integer ceiling of a number (the order of an expansion; SymPy
    passes rational orders, which the classes round up)."""
    from sympy_extras.functions.elementary.integers import ceiling
    c = ceiling(e)
    if not isinstance(c, Integer):
        raise TypeError("the order %s is not a number" % (e,))
    return c.p
