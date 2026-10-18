"""Assumptions: ``is_positive``, ``is_integer``, ... on expressions.

A query returns ``True``, ``False`` or ``None`` (unknown), as in SymPy.
The facts of a symbol are those given to its constructor, completed by
a small table of implications (``positive`` implies ``real``, ``nonzero``
and ``nonnegative``, ``integer`` implies ``rational``, ``even`` implies
``integer``, ...) and their contrapositives. The facts of a compound
expression follow from the facts of its arguments by structural rules:
a sum of integers is an integer, a product of positive numbers is
positive, an even power of a nonzero real number is positive, and so on.
The rules are a subset of SymPy's [1]_ chosen to be sound; a question
they do not answer gives ``None``. For the functions of SymPy wrapped by
:class:`~sympy_extras.core.function.SymPyFunction`, SymPy's ``ask`` is
consulted.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Assumptions*,
       https://docs.sympy.org/latest/guides/assumptions.html
"""
from __future__ import annotations

from typing import Optional

from sympy import Q
from sympy.assumptions import ask
from sympy.assumptions.assume import AppliedPredicate, Predicate

from .expr import Expr
from .numbers import (ComplexInfinity, Exp1, Float, ImaginaryUnit, Infinity,
    NegativeInfinity, Number, Pi, Rational)
from .symbol import Symbol

__all__ = ['Truth', 'closure', 'minimal_generators', 'ask_structural', 'is_number', 'PREDICATES']

#: a three valued truth value
Truth = Optional[bool]

#: the predicates of the assumptions system
PREDICATES = ('commutative', 'complex', 'real', 'extended_real', 'imaginary', 'integer',
    'rational', 'irrational', 'positive', 'negative', 'nonnegative', 'nonpositive', 'zero',
    'nonzero', 'even', 'odd', 'prime', 'finite', 'infinite')

#: implications ``(name, value) -> (name, value)``
_RULES: tuple[tuple[str, bool, str, bool], ...] = (
    ('integer', True, 'rational', True),
    ('rational', True, 'real', True),
    ('irrational', True, 'real', True),
    ('irrational', True, 'rational', False),
    ('real', True, 'complex', True),
    ('real', True, 'extended_real', True),
    ('real', True, 'imaginary', False),
    ('real', True, 'finite', True),
    ('imaginary', True, 'complex', True),
    ('imaginary', True, 'real', False),
    ('imaginary', True, 'extended_real', False),
    ('complex', True, 'finite', True),
    ('complex', True, 'commutative', True),
    ('extended_real', True, 'commutative', True),
    ('infinite', True, 'finite', False),
    ('positive', True, 'real', True),
    ('positive', True, 'nonnegative', True),
    ('positive', True, 'nonzero', True),
    ('positive', True, 'negative', False),
    ('positive', True, 'nonpositive', False),
    ('negative', True, 'real', True),
    ('negative', True, 'nonpositive', True),
    ('negative', True, 'nonzero', True),
    ('negative', True, 'positive', False),
    ('negative', True, 'nonnegative', False),
    ('nonnegative', True, 'real', True),
    ('nonnegative', True, 'negative', False),
    ('nonpositive', True, 'real', True),
    ('nonpositive', True, 'positive', False),
    ('zero', True, 'integer', True),
    ('zero', True, 'even', True),
    ('zero', True, 'nonnegative', True),
    ('zero', True, 'nonpositive', True),
    ('zero', True, 'nonzero', False),
    ('zero', True, 'positive', False),
    ('zero', True, 'negative', False),
    ('nonzero', True, 'real', True),
    ('nonzero', True, 'zero', False),
    ('even', True, 'integer', True),
    ('even', True, 'odd', False),
    ('odd', True, 'integer', True),
    ('odd', True, 'even', False),
    ('odd', True, 'nonzero', True),
    ('prime', True, 'integer', True),
    ('prime', True, 'positive', True),
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
    ('integer', True, 'even', False, 'odd', True),
    ('integer', True, 'odd', False, 'even', True),
)


def closure(facts: dict[str, bool]) -> dict[str, bool]:
    """The facts completed by the implications and their contrapositives.

    Examples
    ========

    >>> from sympy_extras.core.assumptions import closure
    >>> c = closure({'positive': True})
    >>> c['real'], c['nonzero'], c['negative'], c['zero']
    (True, True, False, False)
    >>> closure({'integer': True, 'even': False})['odd']
    True
    """
    known = dict(facts)
    changed = True
    while changed:
        changed = False
        for a, va, b, vb in _RULES:
            if known.get(a) == va and b not in known:
                known[b] = vb
                changed = True
            if known.get(b) == (not vb) and a not in known:
                known[a] = not va
                changed = True
        for a, va, b, vb, c, vc in _PAIR_RULES:
            if known.get(a) == va and known.get(b) == vb and c not in known:
                known[c] = vc
                changed = True
    return known


#: the facts of a node, before the closure
Facts = dict[str, bool]

#: the order in which facts are tried as generators of a closed set
_GENERATOR_ORDER = ('positive', 'negative', 'zero', 'prime', 'even', 'odd', 'integer',
    'rational', 'irrational', 'imaginary', 'nonzero', 'nonnegative', 'nonpositive', 'real',
    'extended_real', 'infinite', 'finite', 'complex', 'commutative')


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
        facts.update(rational=True, finite=True, imaginary=False)
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
        else:
            facts['prime'] = False
        return facts
    if isinstance(n, Float):
        value = n.as_fraction()
        facts.update(real=True, finite=True, imaginary=False)
        facts['zero'] = value == 0
        facts['positive'] = value > 0
        facts['negative'] = value < 0
        return facts
    if isinstance(n, Infinity):
        facts.update(extended_real=True, real=False, complex=False, finite=False, infinite=True,
                     positive=True, nonzero=True, zero=False, negative=False, integer=False,
                     rational=False, imaginary=False)
        return facts
    if isinstance(n, NegativeInfinity):
        facts.update(extended_real=True, real=False, complex=False, finite=False, infinite=True,
                     negative=True, nonzero=True, zero=False, positive=False, integer=False,
                     rational=False, imaginary=False)
        return facts
    if isinstance(n, ComplexInfinity):
        facts.update(real=False, extended_real=False, complex=False, finite=False, infinite=True,
                     zero=False, imaginary=False, integer=False, rational=False)
        return facts
    return facts  # nan: nothing is known


def _constant_facts(e: Expr) -> Facts:
    if isinstance(e, (Pi, Exp1)):
        return {'commutative': True, 'real': True, 'positive': True, 'irrational': True,
                'integer': False, 'prime': False}
    if isinstance(e, ImaginaryUnit):
        return {'commutative': True, 'imaginary': True, 'real': False, 'finite': True,
                'zero': False, 'integer': False, 'rational': False, 'extended_real': False}
    return {}


def _all(values: list[Truth]) -> Truth:
    if all(v is True for v in values):
        return True
    if any(v is False for v in values):
        return False
    return None


def _add_facts(args: tuple[Expr, ...]) -> Facts:
    facts: Facts = {}
    sub = [_facts(a) for a in args]
    for name in ('commutative', 'finite', 'integer', 'rational', 'real', 'complex', 'extended_real'):
        values = [s.get(name) for s in sub]
        if all(v is True for v in values):
            facts[name] = True
        elif name == 'commutative' and any(v is False for v in values):
            facts[name] = False
    for name in ('rational', 'integer'):
        values = [s.get(name) for s in sub]
        reals = [s.get('real') for s in sub]
        if all(r is True for r in reals) and values.count(False) == 1 and values.count(True) == len(values) - 1:
            facts[name] = False
    signs = [(s.get('positive'), s.get('negative'), s.get('nonnegative'), s.get('nonpositive'), s.get('zero'))
             for s in sub]
    if all(p is True for p, _, _, _, _ in signs):
        facts['positive'] = True
    elif all(nn is True for _, _, nn, _, _ in signs):
        facts['nonnegative'] = True
        if any(p is True for p, _, _, _, _ in signs):
            facts['positive'] = True
    if all(n is True for _, n, _, _, _ in signs):
        facts['negative'] = True
    elif all(np is True for _, _, _, np, _ in signs):
        facts['nonpositive'] = True
        if any(n is True for _, n, _, _, _ in signs):
            facts['negative'] = True
    if all(z is True for _, _, _, _, z in signs):
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
    return facts


def _mul_facts(args: tuple[Expr, ...]) -> Facts:
    facts: Facts = {}
    sub = [_facts(a) for a in args]
    for name in ('commutative', 'finite', 'integer', 'rational', 'real', 'complex'):
        values = [s.get(name) for s in sub]
        if all(v is True for v in values):
            facts[name] = True
        elif name == 'commutative' and any(v is False for v in values):
            facts[name] = False
    if any(s.get('zero') is True for s in sub) and all(s.get('finite') is not False for s in sub):
        facts['zero'] = True
    if all(s.get('nonzero') is True for s in sub):
        facts['nonzero'] = True
    if all(s.get('real') is True for s in sub):
        if all(s.get('positive') is True or s.get('negative') is True for s in sub):
            negatives = sum(1 for s in sub if s.get('negative') is True)
            facts['positive'] = negatives % 2 == 0
            facts['negative'] = negatives % 2 == 1
        elif all(s.get('nonnegative') is True or s.get('nonpositive') is True for s in sub):
            negatives = sum(1 for s in sub if s.get('nonpositive') is True and s.get('nonnegative') is not True)
            facts['nonnegative'] = negatives % 2 == 0
            facts['nonpositive'] = negatives % 2 == 1
    if facts.get('integer'):
        if any(s.get('even') is True for s in sub):
            facts['even'] = True
        elif all(s.get('odd') is True for s in sub):
            facts['odd'] = True
    imaginary = [s for s in sub if s.get('imaginary') is True]
    if len(imaginary) == 1 and all(s.get('real') is True or s.get('imaginary') is True for s in sub) \
            and all(s.get('zero') is False for s in sub):
        facts.update(imaginary=True, real=False, zero=False, extended_real=False)
        if all(s.get('finite') is True for s in sub):
            facts['finite'] = True
    return facts


def _pow_facts(base: Expr, exp: Expr) -> Facts:
    facts: Facts = {}
    b, e = _facts(base), _facts(exp)
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
    if b.get('positive') and e.get('real'):
        facts.update(positive=True, real=True)
    if b.get('nonnegative') and e.get('positive'):
        facts.update(nonnegative=True, real=True)
    if b.get('real') and b.get('nonzero') and e.get('even'):
        facts['positive'] = True
    if b.get('real') and e.get('even'):
        facts['nonnegative'] = True
    if b.get('negative') and e.get('odd'):
        facts['negative'] = True
    if b.get('zero') and e.get('positive'):
        facts['zero'] = True
    if b.get('nonzero') and e.get('real') and e.get('finite'):
        facts['nonzero'] = True
    if b.get('finite') and e.get('integer') and e.get('nonnegative'):
        facts['finite'] = True
    if b.get('positive') and b.get('finite') and e.get('real') and e.get('finite'):
        facts['finite'] = True
    return facts


_Q: dict[str, Predicate] = {
    'commutative': Q.commutative, 'complex': Q.complex, 'real': Q.real,
    'extended_real': Q.extended_real, 'imaginary': Q.imaginary, 'integer': Q.integer,
    'rational': Q.rational, 'irrational': Q.irrational, 'positive': Q.positive,
    'negative': Q.negative, 'nonnegative': Q.nonnegative, 'nonpositive': Q.nonpositive,
    'zero': Q.zero, 'nonzero': Q.nonzero, 'even': Q.even, 'odd': Q.odd, 'prime': Q.prime,
    'finite': Q.finite, 'infinite': Q.infinite,
}


def _sympy_facts(e: Expr) -> Facts:
    """The facts SymPy's ``ask`` knows about a wrapped function."""
    from .interop import to_sympy
    facts: Facts = {}
    sympy_expr = to_sympy(e)
    for name, predicate in _Q.items():
        applied = predicate(sympy_expr)
        if not isinstance(applied, AppliedPredicate):
            continue
        answer = ask(applied)
        if isinstance(answer, bool):
            facts[name] = answer
    return facts


def _facts(e: Expr) -> Facts:
    from .add import Add
    from .function import AppliedSymPyFunction, Application, Derivative
    from .mul import Mul
    from .power import Pow
    if isinstance(e, Number):
        return closure(_number_facts(e))
    if isinstance(e, Symbol):
        facts = dict(e.given_assumptions)
        facts.setdefault('commutative', True)
        return closure(facts)
    if isinstance(e, Add):
        return closure(_add_facts(e.args))
    if isinstance(e, Mul):
        return closure(_mul_facts(e.args))
    if isinstance(e, Pow):
        return closure(_pow_facts(e.base, e.exp))
    if isinstance(e, AppliedSymPyFunction):
        return closure(_sympy_facts(e))
    if isinstance(e, Application):
        sub = [_facts(a) for a in e.args]
        commutative = _all([s.get('commutative') for s in sub])
        return closure({} if commutative is None else {'commutative': commutative})
    if isinstance(e, Derivative):
        commutative = _facts(e.expr).get('commutative')
        return closure({} if commutative is None else {'commutative': commutative})
    return closure(_constant_facts(e))


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
    return _facts(e).get(name)


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

