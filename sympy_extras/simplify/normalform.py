r"""A normal form for rational expressions in trigonometric functions and
radicals: the quotient of two polynomials in the kernels, reduced modulo
the algebraic relations between them.

An expression is read as a rational function of its *kernels* (the
symbols and the subexpressions which are not sums, products or integer
powers). Taken as independent indeterminates the kernels give the
quotient of expanded polynomials of
:func:`sympy_extras.matrices.linalg.normal`, in which
$\sin^2\theta + \cos^2\theta - 1$ or $\sqrt{x}^2 - x$ are not zero. Here
the kernels are first brought to a set with known relations:

* $\tan u$, $\cot u$, $\sec u$, $\csc u$ are written with $\sin u$ and
  $\cos u$, which satisfy $\cos^2 u = 1 - \sin^2 u$; the hyperbolic
  functions likewise, with $\cosh^2 u = 1 + \sinh^2 u$;
* a power $b^{p/q}$ is written with the root $\rho = d\, b^{1/q}$, where
  $b = n/d$ in lowest terms, which satisfies the polynomial relation
  $\rho^q = n\, d^{q-1}$ (no choice of branch is involved: this is an
  identity of the principal root), so that all the fractional powers of
  the same base and denominator become powers of one kernel;
* the imaginary unit satisfies $i^2 = -1$.

With the roots ordered from the outermost in, then the cosines, then the
sines and the other kernels, these relations have the pairwise coprime
leading monomials $\rho^q$, $\cos^2 u$, $i^2$ in the lexicographic order:
they are a Gröbner basis of the ideal they generate [1]_, and the
remainder of the division by them is a canonical form modulo that ideal.
The numerator and the denominator are replaced by their remainders and
the quotient is cancelled again; the reduction is in fact done after
every operation, so that the denominators met on the way are in one form
and have common factors to find. With the sines before the cosines the
same holds with $\sin^2 u \to 1 - \cos^2 u$: the result is converted to
that form at the end, and the one with fewer terms is returned.

The result is zero exactly when the numerator lies in the ideal of the
relations: a zero test for these expressions as long as the kernels have
no relation beyond the ones listed (``sin(2*x)`` and ``sin(x)`` are
independent kernels here, and so are ``exp(x)`` and ``exp(2*x)``). The
common factors are those of the polynomials: ``(x - 1)/(sqrt(x) - 1)``
is left as it is.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import sin, cos, tan, sqrt
>>> from sympy_extras.simplify.normalform import normal_form
>>> x, y = Symbol('x'), Symbol('y')
>>> normal_form(sin(x)**4 - cos(x)**4 + cos(x)**2 - sin(x)**2)
0
>>> normal_form(1/(1 - sin(x)) + 1/(1 + sin(x)) - 2/cos(x)**2), normal_form(tan(x)*cos(x))
(0, sin(x))
>>> normal_form(x**2/sqrt(x**2 + 1) - sqrt(x**2 + 1) + 1/sqrt(x**2 + 1))
0
>>> normal_form((1 - cos(x)**2)/(sin(x)*(y + sin(x)) - sin(x)**2))
sin(x)/y

References
==========

.. [1] D. Cox, J. Little, D. O'Shea, *Ideals, Varieties, and Algorithms*,
       4th ed., Springer, 2015, Section 2.9 (polynomials with relatively
       prime leading monomials form a Gröbner basis) and Section 2.6
       (remainders as normal forms).
"""
from __future__ import annotations

from typing import Final, Sequence

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, ImaginaryUnit, Integer, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.hyperbolic import Cosh, Coth, Csch, Sech, Sinh, Tanh
from sympy_extras.functions.elementary.trigonometric import Cos, Cot, Csc, Sec, Sin, Tan
from sympy_extras.functions import cos, cosh, sin, sinh
from sympy_extras.matrices.linalg import _IntegerLift, _PolynomialLift, _collect_kernels
from sympy_extras.polys.rings import PolyElement

__all__ = ['normal_form']


def _numer_denom(e: Expr) -> tuple[Expr, Expr]:
    """The numerator and the denominator of the quotient of expanded
    polynomials in the kernels, in lowest terms."""
    found: dict[Expr, int] = {}
    _collect_kernels(e, found)
    if not found:
        integers = _IntegerLift()
        p, q = integers.cancel(*integers.fraction(e))
        return Integer(p), Integer(q)
    lift = _PolynomialLift(list(found))
    n, d = lift.cancel(*lift.fraction(e))
    return lift.expr(n), lift.expr(d)


class _Abstraction:
    """The expression with its kernels replaced by placeholders which
    satisfy polynomial relations."""

    def __init__(self) -> None:
        self.roots: Final[dict[tuple[Expr, int], tuple[Symbol, Expr]]] = {}      # (base, q) -> (rho, d)
        self.root_order: Final[list[Symbol]] = []
        self.cosines: Final[dict[tuple[bool, Expr], Symbol]] = {}                 # (hyperbolic, argument) -> c
        self.sines: Final[dict[tuple[bool, Expr], Symbol]] = {}
        self.relations: Final[list[Expr]] = []
        self.back: Final[dict[Symbol, Expr]] = {}
        self.imaginary: bool = False

    def _pair(self, hyperbolic: bool, argument: Expr) -> tuple[Symbol, Symbol]:
        key = (hyperbolic, argument)
        if key not in self.sines:
            s: Symbol = Dummy('s')
            c: Symbol = Dummy('c')
            self.sines[key], self.cosines[key] = s, c
            if hyperbolic:
                self.back[s], self.back[c] = sinh(argument), cosh(argument)
                self.relations.append(c**2 - s**2 - 1)
            else:
                self.back[s], self.back[c] = sin(argument), cos(argument)
                self.relations.append(c**2 + s**2 - 1)
        return self.sines[key], self.cosines[key]

    def _restore(self, e: Expr) -> Expr:
        for placeholder, original in self.back.items():
            if e.has(placeholder):
                e = e.subs(placeholder, original)
        return e

    def abstract(self, e: Expr) -> Expr:
        if isinstance(e, (Rational, Symbol)):
            return e
        if isinstance(e, ImaginaryUnit):
            self.imaginary = True
            return e
        if isinstance(e, Add):
            return add(*[self.abstract(a) for a in e.args if isinstance(a, Expr)])
        if isinstance(e, Mul):
            return mul(*[self.abstract(a) for a in e.args if isinstance(a, Expr)])
        if isinstance(e, Pow):
            exponent = e.exp
            if isinstance(exponent, Integer):
                return pow(self.abstract(e.base), exponent)
            if isinstance(exponent, Rational):
                p, q = exponent.p, exponent.q
                key = (e.base, q)
                if key not in self.roots:
                    numerator, denominator = _numer_denom(self.abstract(e.base))
                    rho: Symbol = Dummy('rho')
                    self.roots[key] = (rho, denominator)
                    self.root_order.append(rho)
                    self.relations.append(rho**q - numerator*denominator**(q - 1))
                    self.back[rho] = self._restore(denominator)*pow(e.base, Rational(1, q))
                rho, denominator = self.roots[key]
                return pow(rho, Integer(p))/pow(denominator, Integer(p))
            return e
        if isinstance(e, (Sin, Cos, Tan, Cot, Sec, Csc)):
            argument = e.args[0]
            if isinstance(argument, Expr):
                s, c = self._pair(False, argument)
                if isinstance(e, Sin):
                    return s
                if isinstance(e, Cos):
                    return c
                if isinstance(e, Tan):
                    return s/c
                if isinstance(e, Cot):
                    return c/s
                return 1/c if isinstance(e, Sec) else 1/s
        if isinstance(e, (Sinh, Cosh, Tanh, Coth, Sech, Csch)):
            argument = e.args[0]
            if isinstance(argument, Expr):
                s, c = self._pair(True, argument)
                if isinstance(e, Sinh):
                    return s
                if isinstance(e, Cosh):
                    return c
                if isinstance(e, Tanh):
                    return s/c
                if isinstance(e, Coth):
                    return c/s
                return 1/c if isinstance(e, Sech) else 1/s
        return e


class _QuotientLift(_PolynomialLift):
    """Fractions of polynomials in the kernels, reduced modulo the
    relations after every operation."""

    def __init__(self, kernels: Sequence[Expr]) -> None:
        super().__init__(kernels)
        self.divisors: list[PolyElement[int]] = []

    def fraction(self, e: Expr) -> tuple[PolyElement[int], PolyElement[int]]:
        n, d = super().fraction(e)
        if self.divisors and isinstance(e, (Add, Mul, Pow)):
            n, d = n.rem(self.divisors), d.rem(self.divisors)
            if not d:
                raise ZeroDivisionError('division by zero in %s' % (e,))
        return n, d


def _lift(abstraction: _Abstraction, found: dict[Expr, int], sines_first: bool) -> _QuotientLift:
    special: list[Expr] = list(reversed(abstraction.root_order))
    if abstraction.imaginary:
        special.append(I)
    if sines_first:
        special.extend(abstraction.sines.values())
        special.extend(abstraction.cosines.values())
    else:
        special.extend(abstraction.cosines.values())
        special.extend(abstraction.sines.values())
    chosen = set(special)
    lift = _QuotientLift(special + [k for k in found if k not in chosen])
    divisors = [_PolynomialLift.fraction(lift, relation)[0] for relation in abstraction.relations]
    if abstraction.imaginary:
        unit = lift.index[I]
        divisors.append(unit*unit + lift.ring.one)
    lift.divisors = divisors
    return lift


def _moved(f: PolyElement[int], source: _QuotientLift, target: _QuotientLift) -> PolyElement[int]:
    """The polynomial in the ring of ``target`` (the same kernels in
    another order), reduced there."""
    positions = [source.kernels.index(kernel) for kernel in target.kernels]
    terms = {tuple(monomial[i] for i in positions): coefficient for monomial, coefficient in f.terms.items()}
    return target.ring.ring.from_dict(terms).rem(target.divisors)


def normal_form(e: ExprLike) -> Expr:
    """The expression as a quotient of two expanded polynomials in its
    kernels, in lowest terms, reduced modulo the relations of sines and
    cosines, of fractional powers and of the imaginary unit. Tangents,
    cotangents, secants and cosecants come out as sines and cosines.

    >>> from sympy_extras.core import I, Symbol
    >>> from sympy_extras.functions import cosh, sinh, sqrt, tan, cos
    >>> from sympy_extras.simplify.normalform import normal_form
    >>> x = Symbol('x')
    >>> normal_form(cosh(x)**2 - sinh(x)**2), normal_form((1 + I)**2/(1 - I)), normal_form(1 + tan(x)**2)
    (1, -1 + I, cos(x)**(-2))
    >>> normal_form((sqrt(x) + 1)*(sqrt(x) - 1)), normal_form(sqrt(x)**3/x + 1/sqrt(x))
    (x - 1, (x + 1)/sqrt(x))
    """
    expression = to_expr(e)
    abstraction = _Abstraction()
    abstracted = abstraction.abstract(expression)
    found: dict[Expr, int] = {}
    _collect_kernels(abstracted, found)
    relations = abstraction.relations
    for relation in relations:
        _collect_kernels(relation, found)
    if not found:
        integers = _IntegerLift()
        return integers.ratio(*integers.fraction(abstracted))
    # the arithmetic is done modulo the relations, with cos**2 -> 1 - sin**2: the denominators of the terms of a
    # sum are then in one form, and their least common multiple is found
    lift = _lift(abstraction, found, False)
    numerator, denominator = lift.cancel(*lift.fraction(abstracted))
    best = lift.ratio(numerator, denominator)
    if abstraction.sines and len(numerator.terms) + len(denominator.terms) > 2:
        # sin**2 -> 1 - cos**2 gives another normal form: the one with fewer terms is returned
        other = _lift(abstraction, found, True)
        p, q = _moved(numerator, lift, other), _moved(denominator, lift, other)
        if not q:
            raise ZeroDivisionError('division by zero in %s' % (expression,))
        p, q = other.cancel(p, q)
        if len(p.terms) + len(q.terms) < len(numerator.terms) + len(denominator.terms):
            best = other.ratio(p, q)
    return abstraction._restore(best)
