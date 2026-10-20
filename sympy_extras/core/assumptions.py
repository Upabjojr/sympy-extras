"""Assumptions: ``is_positive``, ``is_integer``, ... on expressions.

A query returns ``True``, ``False`` or ``None`` (unknown), as in SymPy.
The facts of a symbol are those given to its constructor, completed by
a table of implications (``positive`` implies ``real``, ``nonzero``
and ``nonnegative``, ``integer`` implies ``rational`` and ``algebraic``,
``even`` implies ``integer``, ``extended_positive`` and ``finite`` imply
``positive``, ...) and their contrapositives. The facts of a compound
expression follow from the facts of its arguments by structural rules:
a sum of integers is an integer, a product of positive numbers is
positive, an even power of a nonzero real number is positive, a sum of
algebraic numbers and one transcendental number is transcendental, and
so on. The rules are a subset of SymPy's [1]_ chosen to be sound; a
question they do not answer gives ``None``. For the functions of SymPy
wrapped by :class:`~sympy_extras.core.function.SymPyFunction`, SymPy's own
``is_*`` properties are consulted. The facts of a node are computed once and cached
on the node (the tree is immutable).

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Assumptions*,
       https://docs.sympy.org/latest/guides/assumptions.html
"""
from __future__ import annotations

import mpmath

from typing import Callable, Iterable, Optional

from sympy.core.basic import Basic as SymPyBasic

from .expr import Expr
from .numbers import (CatalanType, ComplexInfinity, EulerGammaType, Exp1, Float, GoldenRatioType,
    ImaginaryUnit, Infinity, NegativeInfinity, Number, Pi, Rational, TribonacciConstantType)
from .symbol import Symbol

__all__ = ['Truth', 'Facts', 'closure', 'minimal_generators', 'ask_structural', 'is_number',
    'PREDICATES', 'facts_of', 'constant_facts', 'add_facts', 'mul_facts', 'pow_facts',
    'check_assumptions', 'failing_assumptions', 'common_assumptions', 'assumptions']

#: a three valued truth value
Truth = Optional[bool]
#: the facts of a node
Facts = dict[str, bool]

#: the predicates of the assumptions system
PREDICATES = ('commutative', 'complex', 'real', 'extended_real', 'imaginary', 'integer',
    'noninteger', 'rational', 'irrational', 'algebraic', 'transcendental', 'positive', 'negative',
    'nonnegative', 'nonpositive', 'zero', 'nonzero', 'extended_positive', 'extended_negative',
    'extended_nonnegative', 'extended_nonpositive', 'extended_nonzero', 'even', 'odd', 'prime',
    'composite', 'finite', 'infinite', 'hermitian', 'antihermitian', 'polar')

#: implications ``(name, value) -> (name, value)``
_RULES: tuple[tuple[str, bool, str, bool], ...] = (
    ('integer', True, 'rational', True),
    ('integer', True, 'noninteger', False),
    ('rational', True, 'real', True),
    ('rational', True, 'algebraic', True),
    ('irrational', True, 'real', True),
    ('irrational', True, 'rational', False),
    ('real', True, 'complex', True),
    ('real', True, 'extended_real', True),
    ('real', True, 'imaginary', False),
    ('real', True, 'finite', True),
    ('real', True, 'hermitian', True),
    ('imaginary', True, 'complex', True),
    ('imaginary', True, 'real', False),
    ('imaginary', True, 'extended_real', False),
    ('imaginary', True, 'antihermitian', True),
    ('complex', True, 'finite', True),
    ('complex', True, 'commutative', True),
    ('extended_real', True, 'commutative', True),
    ('algebraic', True, 'complex', True),
    ('algebraic', True, 'transcendental', False),
    ('transcendental', True, 'complex', True),
    ('infinite', True, 'finite', False),
    ('finite', False, 'infinite', True),
    ('positive', True, 'real', True),
    ('positive', True, 'nonnegative', True),
    ('positive', True, 'nonzero', True),
    ('positive', True, 'negative', False),
    ('positive', True, 'nonpositive', False),
    ('positive', True, 'extended_positive', True),
    ('negative', True, 'real', True),
    ('negative', True, 'nonpositive', True),
    ('negative', True, 'nonzero', True),
    ('negative', True, 'positive', False),
    ('negative', True, 'nonnegative', False),
    ('negative', True, 'extended_negative', True),
    ('nonnegative', True, 'real', True),
    ('nonnegative', True, 'negative', False),
    ('nonnegative', True, 'extended_nonnegative', True),
    ('nonpositive', True, 'real', True),
    ('nonpositive', True, 'positive', False),
    ('nonpositive', True, 'extended_nonpositive', True),
    ('zero', True, 'integer', True),
    ('zero', True, 'even', True),
    ('zero', True, 'nonnegative', True),
    ('zero', True, 'nonpositive', True),
    ('zero', True, 'nonzero', False),
    ('zero', True, 'extended_nonzero', False),
    ('zero', True, 'positive', False),
    ('zero', True, 'negative', False),
    ('nonzero', True, 'real', True),
    ('nonzero', True, 'zero', False),
    ('nonzero', True, 'extended_nonzero', True),
    ('extended_positive', True, 'extended_nonnegative', True),
    ('extended_positive', True, 'extended_nonzero', True),
    ('extended_positive', True, 'extended_negative', False),
    ('extended_positive', True, 'extended_nonpositive', False),
    ('extended_negative', True, 'extended_nonpositive', True),
    ('extended_negative', True, 'extended_nonzero', True),
    ('extended_negative', True, 'extended_positive', False),
    ('extended_negative', True, 'extended_nonnegative', False),
    ('extended_nonnegative', True, 'extended_real', True),
    ('extended_nonnegative', True, 'extended_negative', False),
    ('extended_nonpositive', True, 'extended_real', True),
    ('extended_nonpositive', True, 'extended_positive', False),
    ('extended_nonzero', True, 'extended_real', True),
    ('extended_nonzero', True, 'zero', False),
    ('noninteger', True, 'extended_real', True),
    ('noninteger', True, 'integer', False),
    ('even', True, 'integer', True),
    ('even', True, 'odd', False),
    ('odd', True, 'integer', True),
    ('odd', True, 'even', False),
    ('prime', True, 'integer', True),
    ('prime', True, 'positive', True),
    ('prime', True, 'composite', False),
    ('composite', True, 'integer', True),
    ('composite', True, 'positive', True),
    ('composite', True, 'prime', False),
)

#: implications with two premises: ``(a, va, b, vb) -> (c, vc)``
_PAIR_RULES: tuple[tuple[str, bool, str, bool, str, bool], ...] = (
    ('nonnegative', True, 'nonzero', True, 'positive', True),
    ('nonpositive', True, 'nonzero', True, 'negative', True),
    ('nonnegative', True, 'nonpositive', True, 'zero', True),
    ('real', True, 'negative', False, 'nonnegative', True),
    ('real', True, 'positive', False, 'nonpositive', True),
    ('real', True, 'zero', False, 'nonzero', True),
    ('real', True, 'nonzero', False, 'zero', True),
    ('real', True, 'rational', False, 'irrational', True),
    ('integer', True, 'even', False, 'odd', True),
    ('integer', True, 'odd', False, 'even', True),
    ('extended_nonnegative', True, 'extended_nonzero', True, 'extended_positive', True),
    ('extended_nonpositive', True, 'extended_nonzero', True, 'extended_negative', True),
    ('extended_nonnegative', True, 'extended_nonpositive', True, 'zero', True),
    ('extended_real', True, 'extended_negative', False, 'extended_nonnegative', True),
    ('extended_real', True, 'extended_positive', False, 'extended_nonpositive', True),
    ('extended_real', True, 'zero', False, 'extended_nonzero', True),
    ('extended_real', True, 'extended_nonzero', False, 'zero', True),
    ('extended_real', True, 'integer', False, 'noninteger', True),
    ('extended_real', True, 'finite', True, 'real', True),
    ('extended_positive', True, 'finite', True, 'positive', True),
    ('extended_negative', True, 'finite', True, 'negative', True),
    ('extended_nonnegative', True, 'finite', True, 'nonnegative', True),
    ('extended_nonpositive', True, 'finite', True, 'nonpositive', True),
    ('extended_nonzero', True, 'finite', True, 'nonzero', True),
    ('complex', True, 'algebraic', False, 'transcendental', True),
)


def closure(facts: dict[str, bool]) -> dict[str, bool]:
    """The facts completed by the implications and their contrapositives.

    Examples
    ========

    >>> from sympy_extras.core.assumptions import closure
    >>> c = closure({'positive': True})
    >>> c['real'], c['nonzero'], c['negative'], c['zero'], c['extended_positive']
    (True, True, False, False, True)
    >>> closure({'integer': True, 'even': False})['odd']
    True
    """
    known = dict(facts)  # the given facts first, so that a contradiction among them stays visible
    pending = list(facts.items())
    while pending:
        for name, value in pending:
            if known.get(name) == value:
                for b, vb in _IMPLIED[(name, value)].items():
                    known.setdefault(b, vb)
        pending = []
        for a, va, b, vb, c, vc in _PAIR_RULES:
            if c not in known and known.get(a) == va and known.get(b) == vb:
                known[c] = vc
                pending.append((c, vc))
    return known


def _implied(name: str, value: bool) -> dict[str, bool]:
    """Every fact which follows from the single fact by the implications
    and their contrapositives (the fact itself first)."""
    found: dict[str, bool] = {name: value}
    pending = [(name, value)]
    while pending:
        a, va = pending.pop()
        for p, vp, q, vq in _RULES:
            if (p, vp) == (a, va) and q not in found:
                found[q] = vq
                pending.append((q, vq))
            if (q, not vq) == (a, va) and p not in found:
                found[p] = not vp
                pending.append((p, not vp))
    return found


#: the consequences of each single fact, computed once (the closure of a
#: set of facts is their union, plus the pair rules)
_IMPLIED: dict[tuple[str, bool], dict[str, bool]] = {
    (name, value): _implied(name, value) for name in PREDICATES for value in (True, False)}


def is_consistent(facts: dict[str, bool]) -> bool:
    """Whether no implication is contradicted by the facts.

    Examples
    ========

    >>> from sympy_extras.core.assumptions import is_consistent
    >>> is_consistent({'positive': True, 'integer': True}), is_consistent({'positive': True, 'zero': True})
    (True, False)
    >>> is_consistent({'even': True, 'odd': True}), is_consistent({'real': True, 'rational': False, 'irrational': False})
    (False, False)
    """
    known = closure(facts)
    for a, va, b, vb in _RULES:
        if known.get(a) == va and known.get(b) == (not vb):
            return False
    for a, va, b, vb, c, vc in _PAIR_RULES:
        if known.get(a) == va and known.get(b) == vb and known.get(c) == (not vc):
            return False
    return True


#: the order in which facts are tried as generators of a closed set
_GENERATOR_ORDER = ('positive', 'negative', 'zero', 'prime', 'composite', 'even', 'odd', 'integer',
    'rational', 'irrational', 'algebraic', 'transcendental', 'imaginary', 'nonzero', 'nonnegative',
    'nonpositive', 'real', 'extended_positive', 'extended_negative', 'extended_nonzero',
    'extended_nonnegative', 'extended_nonpositive', 'noninteger', 'extended_real', 'infinite',
    'finite', 'complex', 'hermitian', 'antihermitian', 'polar', 'commutative')


def minimal_generators(facts: dict[str, bool]) -> dict[str, bool]:
    """A small set of facts whose closure is the closure of ``facts``,
    ``commutative=True`` (the default) left out.

    Examples
    ========

    >>> from sympy_extras.core.assumptions import closure, minimal_generators
    >>> minimal_generators(closure({'positive': True, 'integer': True}))
    {'positive': True, 'integer': True}
    """
    target = closure(facts)
    if target.get('commutative') is True:
        del target['commutative']
    generators: dict[str, bool] = {}
    for name in _GENERATOR_ORDER:
        for value in (True, False):
            if target.get(name) == value and closure(generators).get(name) != value:
                generators[name] = value
    for name, value in target.items():
        if closure(generators).get(name) != value:
            generators[name] = value
    return generators


def _number_facts(n: Number) -> Facts:
    facts: Facts = {'commutative': True}
    if isinstance(n, Rational):
        facts.update(rational=True, finite=True, imaginary=False, algebraic=True)
        facts['integer'] = n.q == 1
        facts['zero'] = n.p == 0
        facts['positive'] = n.p > 0
        facts['negative'] = n.p < 0
        facts['irrational'] = False
        if n.q == 1:
            facts['even'] = n.p % 2 == 0
            facts['odd'] = n.p % 2 == 1
            from sympy import isprime
            facts['prime'] = bool(isprime(n.p))
            facts['composite'] = n.p > 1 and not facts['prime']
        else:
            facts['prime'] = False
            facts['composite'] = False
        return facts
    if isinstance(n, Float):
        value = n.as_fraction()
        facts.update(real=True, finite=True, imaginary=False, composite=False)
        facts['zero'] = value == 0
        facts['positive'] = value > 0
        facts['negative'] = value < 0
        facts['noninteger'] = value.denominator != 1
        return facts
    if isinstance(n, Infinity):
        facts.update(extended_real=True, real=False, complex=False, finite=False, infinite=True,
                     extended_positive=True, zero=False, negative=False, integer=False,
                     rational=False, imaginary=False, algebraic=False, transcendental=False)
        return facts
    if isinstance(n, NegativeInfinity):
        facts.update(extended_real=True, real=False, complex=False, finite=False, infinite=True,
                     extended_negative=True, zero=False, positive=False, integer=False,
                     rational=False, imaginary=False, algebraic=False, transcendental=False)
        return facts
    if isinstance(n, ComplexInfinity):
        facts.update(real=False, extended_real=False, complex=False, finite=False, infinite=True,
                     zero=False, imaginary=False, integer=False, rational=False, algebraic=False,
                     transcendental=False, noninteger=False)
        return facts
    return facts  # nan: nothing is known


def constant_facts(e: Expr) -> Facts:
    """The facts of a number or a named constant (``pi``, ``E``, ``I``, ...),
    empty for anything else."""
    if isinstance(e, Number):
        return _number_facts(e)
    if isinstance(e, (Pi, Exp1)):
        return {'commutative': True, 'real': True, 'positive': True, 'irrational': True,
                'integer': False, 'prime': False, 'transcendental': True, 'composite': False}
    if isinstance(e, (GoldenRatioType, TribonacciConstantType)):
        return {'commutative': True, 'real': True, 'positive': True, 'irrational': True,
                'integer': False, 'prime': False, 'algebraic': True, 'composite': False}
    if isinstance(e, (EulerGammaType, CatalanType)):
        return {'commutative': True, 'real': True, 'positive': True}
    if isinstance(e, ImaginaryUnit):
        return {'commutative': True, 'imaginary': True, 'real': False, 'finite': True,
                'zero': False, 'integer': False, 'rational': False, 'extended_real': False,
                'algebraic': True}
    return {}


def _all(values: list[Truth]) -> Truth:
    if all(v is True for v in values):
        return True
    if any(v is False for v in values):
        return False
    return None


def _sign_facts(sub: list[Facts], prefix: str, facts: Facts) -> None:
    """The sign of a sum from the signs of its terms (``prefix`` selects
    the finite or the extended predicates)."""
    positive, negative = prefix + 'positive', prefix + 'negative'
    nonnegative, nonpositive = prefix + 'nonnegative', prefix + 'nonpositive'
    if all(s.get(positive) is True for s in sub):
        facts[positive] = True
    elif all(s.get(nonnegative) is True for s in sub):
        facts[nonnegative] = True
        if any(s.get(positive) is True for s in sub):
            facts[positive] = True
    if all(s.get(negative) is True for s in sub):
        facts[negative] = True
    elif all(s.get(nonpositive) is True for s in sub):
        facts[nonpositive] = True
        if any(s.get(negative) is True for s in sub):
            facts[negative] = True


def add_facts(args: tuple[Expr, ...]) -> Facts:
    """The facts of a sum from the facts of its terms."""
    facts: Facts = {}
    sub = [facts_of(a) for a in args]
    for name in ('commutative', 'finite', 'integer', 'rational', 'real', 'complex', 'extended_real',
                 'hermitian', 'antihermitian', 'algebraic'):
        values = [s.get(name) for s in sub]
        if all(v is True for v in values):
            facts[name] = True
        elif name == 'commutative' and any(v is False for v in values):
            facts[name] = False
    for name in ('rational', 'integer', 'algebraic'):
        values = [s.get(name) for s in sub]
        domain = 'complex' if name == 'algebraic' else 'real'
        if all(s.get(domain) is True for s in sub) and values.count(False) == 1 and values.count(True) == len(values) - 1:
            facts[name] = False
    if facts.get('algebraic') is False and sum(1 for s in sub if s.get('transcendental') is True) == 1:
        facts['transcendental'] = True
    if facts.get('integer') is False and all(s.get('extended_real') is True for s in sub) \
            and sum(1 for s in sub if s.get('noninteger') is True) == 1:
        facts['noninteger'] = True
    _sign_facts(sub, '', facts)
    _sign_facts(sub, 'extended_', facts)
    if all(s.get('zero') is True for s in sub):
        facts['zero'] = True
    parities = [(s.get('even'), s.get('odd')) for s in sub]
    if facts.get('integer') and all(e is True or o is True for e, o in parities):
        odd = sum(1 for _, o in parities if o is True)
        facts['odd'] = odd % 2 == 1
        facts['even'] = odd % 2 == 0
    # a real part and a nonzero imaginary part: not real, not zero
    imaginary = [s for s in sub if s.get('imaginary') is True]
    if imaginary and all(s.get('real') is True or s.get('imaginary') is True for s in sub):
        if len(imaginary) == len(sub):
            facts['imaginary'] = True
        elif len(imaginary) == 1 and imaginary[0].get('zero') is False:
            facts.update(real=False, zero=False, extended_real=False)
            if all(s.get('finite') is True for s in sub):
                facts['finite'] = True
            if any(s.get('real') is True and s.get('zero') is False for s in sub):
                facts['imaginary'] = False
    # a real part and a single term which is not real: not real, not zero
    non_real = [s for s in sub if s.get('real') is False and s.get('complex') is True]
    if len(non_real) == 1 and all(s.get('real') is True or s is non_real[0] for s in sub):
        facts.update(real=False, extended_real=False, zero=False)
    if 'positive' not in facts and 'negative' not in facts and all(s.get('real') is True for s in sub):
        numbers = [a for a in args if is_number(a)]
        others = [s for a, s in zip(args, sub) if not is_number(a)]
        sign = _numeric_sign(tuple(numbers)) if numbers else None
        if sign is not None and not others:
            facts.update(positive=sign > 0, negative=sign < 0, zero=False)
        elif sign is not None and sign > 0 and all(s.get('nonnegative') is True for s in others):
            facts.update(positive=True, zero=False)  # -p - E + 1/2 is negative, as SymPy knows
        elif sign is not None and sign < 0 and all(s.get('nonpositive') is True for s in others):
            facts.update(negative=True, zero=False)
    # the real terms are numbers: the sum is not imaginary (SymPy answers
    # None when a symbol is among them)
    if facts.get('imaginary') is False and not all(is_number(a) for a, s in zip(args, sub) if s.get('real') is True):
        del facts['imaginary']
    return facts


def numeric_facts(e: Expr) -> Facts:
    """The facts of a numerical expression found by evaluating it at two
    precisions, as SymPy does for numbers: real or not, its sign, purely
    imaginary; nothing when the value is too small to be trusted.

    Examples
    ========

    >>> from sympy_extras.core import I, pi, sympify
    >>> from sympy_extras.core.assumptions import numeric_facts
    >>> numeric_facts((-1 - I)**-2), numeric_facts(pi - 3)
    ({'real': False, 'zero': False, 'imaginary': True}, {'real': True, 'zero': False, 'positive': True, 'negative': False})
    """
    from .evalf import numeric_value
    facts: Facts = {}
    reals: list[int] = []
    imaginaries: list[int] = []
    for prec in (53, 160):
        with mpmath.workprec(prec):
            value = numeric_value(e, prec)
            if value is None:
                return {}
            scale = max(abs(value.real), abs(value.imag))
            if scale == 0:
                return {}
            threshold = scale * mpmath.mpf(2) ** (-(prec - 20))
            reals.append(0 if abs(value.real) <= threshold else (1 if value.real > 0 else -1))
            imaginaries.append(0 if abs(value.imag) <= threshold else (1 if value.imag > 0 else -1))
    if reals[0] != reals[1] or imaginaries[0] != imaginaries[1]:
        return {}
    if imaginaries[0] != 0:
        facts.update(real=False, zero=False)
        if reals[0] == 0:
            facts['imaginary'] = True
    elif reals[0] != 0:
        facts.update(real=True, zero=False, positive=reals[0] > 0, negative=reals[0] < 0)
    return facts


def _numeric_sign(args: tuple[Expr, ...]) -> Optional[int]:
    """The sign of a sum of real numbers found numerically at two
    precisions, ``None`` when the sum is too small to be trusted (an
    exact zero leaves noise)."""
    from .evalf import numeric_value
    signs: list[int] = []
    for prec in (53, 160):
        total = mpmath.mpf(0)
        scale = mpmath.mpf(0)
        with mpmath.workprec(prec):
            for a in args:
                value = numeric_value(a, prec)
                if value is None or value.imag != 0:
                    return None
                total += value.real
                scale = max(scale, abs(value.real))
            if scale == 0 or abs(total) <= scale * mpmath.mpf(2) ** (-(prec - 20)):
                return None
        signs.append(1 if total > 0 else -1)
    return signs[0] if signs[0] == signs[1] else None


def _consecutive_factors(args: tuple[Expr, ...]) -> bool:
    """Whether two factors differ by an odd integer (``n*(n - 1)`` is even)."""
    from .add import sub
    for i, f in enumerate(args):
        for g in args[i + 1:]:
            d = sub(g, f)
            if isinstance(d, Rational) and d.q == 1 and d.p % 2 == 1:
                return True
    return False


def mul_facts(args: tuple[Expr, ...]) -> Facts:
    """The facts of a product from the facts of its factors."""
    facts: Facts = {}
    sub = [facts_of(a) for a in args]
    for name in ('commutative', 'finite', 'integer', 'rational', 'real', 'complex', 'extended_real', 'algebraic'):
        values = [s.get(name) for s in sub]
        if all(v is True for v in values):
            facts[name] = True
        elif name == 'commutative' and any(v is False for v in values):
            facts[name] = False
    if any(s.get('zero') is True for s in sub) and all(s.get('finite') is not False for s in sub):
        facts['zero'] = True
    if all(s.get('nonzero') is True for s in sub):
        facts['nonzero'] = True
    if all(s.get('extended_nonzero') is True for s in sub):
        facts['extended_nonzero'] = True
    if all(s.get('zero') is False for s in sub):
        facts['zero'] = False
    for prefix in ('', 'extended_'):
        positive, negative = prefix + 'positive', prefix + 'negative'
        nonnegative, nonpositive = prefix + 'nonnegative', prefix + 'nonpositive'
        if all(s.get(prefix + 'real' if prefix else 'real') is True for s in sub):
            if all(s.get(positive) is True or s.get(negative) is True for s in sub):
                negatives = sum(1 for s in sub if s.get(negative) is True)
                facts[positive] = negatives % 2 == 0
                facts[negative] = negatives % 2 == 1
            elif all(s.get(nonnegative) is True or s.get(nonpositive) is True for s in sub):
                negatives = sum(1 for s in sub if s.get(nonpositive) is True and s.get(nonnegative) is not True)
                facts[nonnegative if negatives % 2 == 0 else nonpositive] = True  # the product may be zero
    if facts.get('integer'):
        if any(s.get('even') is True for s in sub) or _consecutive_factors(args):
            facts['even'] = True
        elif all(s.get('odd') is True for s in sub):
            facts['odd'] = True
    irrationals = [s for s in sub if s.get('irrational') is True]
    if len(irrationals) == 1 and all(s is irrationals[0] or (s.get('rational') is True and s.get('zero') is False)
                                     for s in sub):
        facts.update(irrational=True, rational=False)  # 3**(2/3)/3
    elif len(args) == 2 and isinstance(args[0], Rational) and args[0].p == 1 and args[0].q == 2:
        # SymPy: an even number halved is an integer, an odd one is not
        if sub[1].get('even') is True:
            facts['integer'] = True
        elif sub[1].get('odd') is True:
            facts.update(integer=False, rational=True)
    imaginary = [s for s in sub if s.get('imaginary') is True]
    if len(imaginary) == 1 and all(s.get('real') is True or s.get('imaginary') is True for s in sub) \
            and all(s.get('zero') is False for s in sub):
        facts.update(imaginary=True, real=False, zero=False, extended_real=False)
        if all(s.get('finite') is True for s in sub):
            facts['finite'] = True
    transcendental = [s for s in sub if s.get('transcendental') is True]
    if len(transcendental) == 1 and all(s.get('algebraic') is True and s.get('zero') is False
                                        for s in sub if s.get('transcendental') is not True):
        facts.update(transcendental=True, algebraic=False)
    # SymPy: a product is polar when it has a polar factor and every
    # factor is polar or positive, and not polar otherwise
    facts['polar'] = any(s.get('polar') is True for s in sub) and \
        all(s.get('polar') is True or s.get('positive') is True for s in sub)
    if 'real' not in facts and all(is_number(a) for a in args):
        from .mul import Mul
        for name, known in numeric_facts(Mul(*args)).items():
            facts.setdefault(name, known)
    # nonzero real factors and a single factor which is not real: not real
    non_real = [s for s in sub if s.get('real') is False and s.get('complex') is True]
    if len(non_real) == 1 and all((s.get('real') is True and s.get('zero') is False) or s is non_real[0] for s in sub):
        facts.update(real=False, extended_real=False)
        if non_real[0].get('zero') is False:
            facts['zero'] = False
        if non_real[0].get('imaginary') is False:
            facts['imaginary'] = False
    return facts


def pow_facts(base: Expr, exp: Expr) -> Facts:
    """The facts of a power from the facts of its base and exponent."""
    facts: Facts = {}
    b, e = facts_of(base), facts_of(exp)
    if b.get('commutative') is True and e.get('commutative') is True:
        facts['commutative'] = True
    elif b.get('commutative') is False or e.get('commutative') is False:
        facts['commutative'] = False
    if b.get('integer') and e.get('integer') and e.get('nonnegative'):
        facts['integer'] = True
    if b.get('rational') and e.get('integer') and (e.get('nonnegative') or b.get('nonzero')):
        facts['rational'] = True
    if b.get('real') and e.get('integer') and (e.get('nonnegative') or b.get('nonzero')):
        facts['real'] = True
    if b.get('extended_real') and e.get('integer') and e.get('positive'):
        facts['extended_real'] = True
    if b.get('positive') and e.get('real'):
        facts.update(positive=True, real=True)
    if b.get('extended_positive') and e.get('positive'):
        facts['extended_positive'] = True
    if b.get('extended_positive') and e.get('real'):
        facts['extended_nonnegative'] = True
    if b.get('nonnegative') and e.get('positive'):
        facts.update(nonnegative=True, real=True)
    if b.get('real') and b.get('nonzero') and e.get('even'):
        facts['positive'] = True
    if b.get('real') and e.get('even') and (e.get('nonnegative') or b.get('nonzero')):
        facts['nonnegative'] = True  # x**(-2) may be zoo
    if b.get('negative') and e.get('odd'):
        facts['negative'] = True
    if b.get('zero') and e.get('positive'):
        facts['zero'] = True
    if b.get('nonzero') and e.get('real') and e.get('finite'):
        facts['zero'] = False  # nonzero (real and not zero) only when the power is real: (-2)**p is not
    if b.get('finite') and e.get('negative') and e.get('finite'):
        facts['zero'] = False  # 1/(p - 4) is never zero
    if b.get('real') and e.get('integer'):
        facts['imaginary'] = False
    if b.get('real') is False and b.get('complex') is True and isinstance(exp, Rational) and exp.q == 1 \
            and abs(exp.p) == 1:
        facts.update(real=False, extended_real=False)  # 1/(pi + I)
    if 'real' not in facts and is_number(base) and is_number(exp):
        from .power import Pow
        for name, known in numeric_facts(Pow(base, exp)).items():
            facts.setdefault(name, known)
    if b.get('finite') and e.get('integer') and e.get('nonnegative'):
        facts['finite'] = True
    if b.get('finite') and e.get('positive'):
        facts['finite'] = True  # sqrt(log(n))
    if b.get('positive') and b.get('finite') and e.get('finite'):
        facts['finite'] = True  # p**I
    if b.get('finite') and b.get('zero') is False and e.get('real') and e.get('finite'):
        facts['finite'] = True  # 1/sqrt(p + I)
    if b.get('algebraic') and isinstance(exp, Rational) and (exp.p >= 0 or b.get('zero') is False):
        facts['algebraic'] = True
    if b.get('real') is False and b.get('complex') is True and isinstance(exp, Rational) and abs(exp.p) == 1 \
            and exp.q > 1:
        facts.update(real=False, extended_real=False)  # a root of a non-real number is not real
    if b.get('transcendental') and isinstance(exp, Rational) and exp.p != 0:
        facts.update(transcendental=True, algebraic=False)
    if b.get('polar'):
        facts['polar'] = True
    if isinstance(base, Rational) and base.p > 0 and isinstance(exp, Rational) and exp.q > 1:
        # a non-integer power of a positive rational which is not a
        # perfect power is irrational (the node is kept when it is not)
        from .power import Pow, pow
        value = pow(base, exp)
        if isinstance(value, Pow) and value.base == base and value.exp == exp:
            facts.update(irrational=True, real=True)
    return facts


#: SymPy's ``is_*`` properties by predicate name (read without ``getattr``)
_SYMPY_QUERIES: dict[str, Callable[[SymPyBasic], Truth]] = {
    'commutative': lambda e: e.is_commutative, 'complex': lambda e: e.is_complex, 'real': lambda e: e.is_real,
    'extended_real': lambda e: e.is_extended_real, 'imaginary': lambda e: e.is_imaginary,
    'integer': lambda e: e.is_integer, 'noninteger': lambda e: e.is_noninteger, 'rational': lambda e: e.is_rational,
    'irrational': lambda e: e.is_irrational, 'algebraic': lambda e: e.is_algebraic,
    'transcendental': lambda e: e.is_transcendental, 'positive': lambda e: e.is_positive,
    'negative': lambda e: e.is_negative, 'nonnegative': lambda e: e.is_nonnegative,
    'nonpositive': lambda e: e.is_nonpositive, 'zero': lambda e: e.is_zero, 'nonzero': lambda e: e.is_nonzero,
    'extended_positive': lambda e: e.is_extended_positive, 'extended_negative': lambda e: e.is_extended_negative,
    'extended_nonnegative': lambda e: e.is_extended_nonnegative,
    'extended_nonpositive': lambda e: e.is_extended_nonpositive, 'extended_nonzero': lambda e: e.is_extended_nonzero,
    'even': lambda e: e.is_even, 'odd': lambda e: e.is_odd, 'prime': lambda e: e.is_prime,
    'composite': lambda e: e.is_composite, 'finite': lambda e: e.is_finite, 'infinite': lambda e: e.is_infinite,
    'hermitian': lambda e: e.is_hermitian, 'antihermitian': lambda e: e.is_antihermitian, 'polar': lambda e: e.is_polar,
}


def sympy_facts(e: Expr) -> Facts:
    """The facts SymPy knows about the SymPy form of ``e`` (its ``is_*``
    properties)."""
    from .interop import to_sympy
    facts: Facts = {}
    sympy_expr = to_sympy(e)
    for name, query in _SYMPY_QUERIES.items():
        try:
            answer = query(sympy_expr)
        except (TypeError, ValueError, AttributeError):  # SymPy refuses some questions
            continue
        if isinstance(answer, bool):
            facts[name] = answer
    return facts


def facts_of(e: Expr) -> Facts:
    """The closed facts of ``e``, computed once and cached on the node.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.core.assumptions import facts_of
    >>> p = Symbol('p', positive=True)
    >>> facts_of(p + 1)['positive'], facts_of(p + 1)['zero']
    (True, False)
    """
    cached = e._facts_cache
    if cached is None:
        cached = closure(e._eval_facts())
        e._facts_cache = cached
    return cached


def ask_structural(e: Expr, name: str) -> Truth:
    """``True``, ``False`` or ``None`` for the predicate ``name`` on ``e``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.core.assumptions import ask_structural
    >>> n = Symbol('n', integer=True)
    >>> ask_structural(n**2 + 1, 'positive'), ask_structural(2*n, 'even'), ask_structural(n, 'positive')
    (True, True, None)
    """
    if name not in PREDICATES:
        raise ValueError("unknown predicate %r" % (name,))
    return facts_of(e).get(name)


def is_number(e: Expr) -> bool:
    """Whether ``e`` is built from numbers and constants only.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, pi, sqrt
    >>> from sympy_extras.core.assumptions import is_number
    >>> x = Symbol('x')
    >>> is_number(2 + pi), is_number(sqrt(2)), is_number(x + 1)
    (True, True, False)
    """
    from .function import AppliedUndef, Derivative
    if isinstance(e, (AppliedUndef, Derivative)):
        return False
    if isinstance(e, Symbol):
        return False
    if not e.args:
        return True
    return all(is_number(a) if isinstance(a, Expr) else False for a in e.args)


def assumptions(expr: Expr, _check: Optional[Iterable[str]] = None) -> dict[str, Truth]:
    """The facts of ``expr`` which are known (``True`` or ``False``), or the
    values of the predicates in ``_check``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.core.assumptions import assumptions
    >>> p = Symbol('p', positive=True)
    >>> assumptions(Symbol('x')), assumptions(p, ['positive', 'real', 'integer'])
    ({'commutative': True}, {'positive': True, 'real': True, 'integer': None})
    """
    facts = facts_of(expr)
    if _check is not None:
        return {name: facts.get(name) for name in _check}
    return {name: value for name, value in facts.items()}


def check_assumptions(expr: Expr, against: Optional[Expr] = None, **assume: Truth) -> Truth:
    """Whether ``expr`` satisfies the given facts (or the known facts of
    ``against``): ``True``, ``False`` when one fails, ``None`` when one is
    unknown.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.core.assumptions import check_assumptions
    >>> p = Symbol('p', positive=True)
    >>> check_assumptions(p, positive=True), check_assumptions(Symbol('x'), positive=True)
    (True, None)
    >>> check_assumptions(Integer(-2), positive=True), check_assumptions(Integer(2), Symbol('n', integer=True))
    (False, True)
    """
    if against is not None:
        if assume:
            raise ValueError("check_assumptions takes either an expression or keyword facts")
        assume = {k: v for k, v in facts_of(against).items()}
    failing = failing_assumptions(expr, **assume)
    if not failing:
        return True
    if any(v is not None for v in failing.values()):
        return False
    return None


def failing_assumptions(expr: Expr, **assume: Truth) -> dict[str, Truth]:
    """The given facts which ``expr`` does not satisfy, with their values on
    ``expr``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.core.assumptions import failing_assumptions
    >>> failing_assumptions(Integer(-2), positive=True, integer=True), failing_assumptions(Symbol('x'), real=True)
    ({'positive': False}, {'real': None})
    """
    facts = facts_of(expr)
    failing: dict[str, Truth] = {}
    for name, value in assume.items():
        if name not in PREDICATES:
            raise ValueError("unknown predicate %r" % (name,))
        actual = facts.get(name)
        if actual != value:
            failing[name] = actual
    return failing


def common_assumptions(exprs: Iterable[Expr], check: Optional[Iterable[str]] = None) -> dict[str, bool]:
    """The facts with the same known value on all ``exprs`` (among those
    in ``check``, when given).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.core.assumptions import common_assumptions
    >>> common_assumptions([Integer(2), Symbol('p', positive=True)], check=['positive', 'zero', 'integer'])
    {'positive': True, 'zero': False}
    """
    all_facts = [facts_of(e) for e in exprs]
    if not all_facts:
        return {}
    names = list(check) if check is not None else list(PREDICATES)
    common: dict[str, bool] = {}
    for name in names:
        values = [f.get(name) for f in all_facts]
        if values[0] is not None and all(v == values[0] for v in values):
            common[name] = values[0]
    return common
