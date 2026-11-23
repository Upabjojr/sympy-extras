r"""The LaTeX rules of the functions which have a notation of their own
(``e^{x}``, ``\left|{x}\right|``, ``J_{n}\left(x\right)``, ...).

A typed port of the function printers of SymPy's ``printing/latex.py``.
SymPy passes the exponent of a power into the printer of its base as a
keyword (``exp=``), and each printer brackets itself; here a rule reads
the exponent offered by the context and renders it on its head when the
notation has a place for it (``\Gamma^{2}\left(x\right)``), and returns
what it printed otherwise: the base of the power is then bracketed where
all brackets are, from the binding of the result.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/printing/latex.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; rules in a table by class; the exponent of the
# parent taken from the context; the defects of sympy/sympy#30844 fixed.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Sequence

from sympy_extras.concrete.qhyper import QPochhammer
from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import Application
from sympy_extras.core.mul import neg
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.functions.combinatorial.factorials import (Binomial, Factorial, Factorial2, FallingFactorial,
    RisingFactorial, Subfactorial)
from sympy_extras.functions.combinatorial.numbers import (Bell, Bernoulli, Catalan, DivisorSigma, Euler, Fibonacci,
    Genocchi, Lucas, Mobius, PrimeNu, PrimeOmega, ReducedTotient, Totient, Tribonacci, UDivisorSigma)
from sympy_extras.functions.elementary.complexes import Abs, Adjoint, Conjugate, Im, Re, Transpose
from sympy_extras.functions.elementary.exponential import Exp, LambertW, Log
from sympy_extras.functions.elementary.hyperbolic import Acosh, Acoth, Acsch, Asech, Asinh, Atanh
from sympy_extras.functions.elementary.integers import Ceiling, Floor, Frac
from sympy_extras.functions.elementary.miscellaneous import Max, Min
from sympy_extras.functions.elementary.trigonometric import Acos, Acot, Acsc, Asec, Asin, Atan
from sympy_extras.functions.special.bessel import (AiryAi, AiryAiPrime, AiryBi, AiryBiPrime, BesselI, BesselJ, BesselK,
    BesselY, Hankel1, Hankel2, SphericalHankel1, SphericalHankel2, SphericalJ, SphericalY)
from sympy_extras.functions.special.beta_functions import Beta, BetaInc, BetaIncRegularized
from sympy_extras.functions.special.delta_functions import DiracDelta, Heaviside
from sympy_extras.functions.special.elliptic_integrals import EllipticE, EllipticF, EllipticK, EllipticPi
from sympy_extras.functions.special.error_functions import ChiNode, Expint, FresnelC, FresnelS
from sympy_extras.functions.special.gamma_functions import Gamma, LowerGamma, UpperGamma
from sympy_extras.functions.special.hyper import Hyper, MeijerG
from sympy_extras.functions.special.mathieu_functions import MathieuC, MathieuCPrime, MathieuS, MathieuSPrime
from sympy_extras.functions.special.polynomials import (AssocLaguerre, AssocLegendre, ChebyshevT, ChebyshevU,
    Gegenbauer, Hermite, Jacobi, Laguerre, Legendre)
from sympy_extras.functions.special.singularity_functions import SingularityFunctionNode
from sympy_extras.functions.special.spherical_harmonics import YnmNode, ZnmNode
from sympy_extras.functions.special.tensor_functions import KroneckerDeltaNode, LeviCivitaNode
from sympy_extras.functions.special.zeta_functions import DirichletEta, LerchPhi, PolyLog, Stieltjes, Zeta

from ._latex_names import ACCEPTED_FUNCTIONS
from .framework import Binding, Result
from .latex import RULES, Ctx, LatexPrinter, Res

__all__: list[str] = []


def _register(classes: Sequence[type[Application]], rule: Callable[[LatexPrinter, Application, Ctx], Res]) -> None:
    """One rule for several classes of functions."""
    for cls in classes:
        RULES.register(cls, rule)


def _round(p: LatexPrinter, args: Sequence[Expr], ctx: Ctx) -> str:
    r"""``\left(a, b\right)``."""
    return r'\left(%s\right)' % p.join(args, ctx)


def _delimited(tex: str) -> Res:
    """A notation which delimits itself (``|x|``, a floor, a binomial)."""
    return Result(tex, Binding.ATOM)


def _applied(tex: str) -> Res:
    """A notation which takes no exponent on its head."""
    return Result(tex, Binding.APPLICATION, applies=True)


# ----------------------------------------------------------------------
# elementary functions


@RULES.of(Exp)
def _exp(p: LatexPrinter, e: Exp, ctx: Ctx) -> Res:
    return Result('e^{%s}' % p.print(e.args[0], ctx.exponent()).output, Binding.POWER)


@RULES.of(Log)
def _log(p: LatexPrinter, e: Log, ctx: Ctx) -> Res:
    head = r'\ln' if p.settings.ln_notation else r'\log'
    return _delimited(r'%s{\left(%s \right)}' % (head, p.inside(e.args[0], ctx)))


@RULES.of(Abs)
def _abs(p: LatexPrinter, e: Abs, ctx: Ctx) -> Res:
    return _delimited(r'\left|{%s}\right|' % p.inside(e.args[0], ctx))


@RULES.of(Floor)
def _floor(p: LatexPrinter, e: Floor, ctx: Ctx) -> Res:
    return _delimited(r'\left\lfloor{%s}\right\rfloor' % p.inside(e.args[0], ctx))


@RULES.of(Ceiling)
def _ceiling(p: LatexPrinter, e: Ceiling, ctx: Ctx) -> Res:
    return _delimited(r'\left\lceil{%s}\right\rceil' % p.inside(e.args[0], ctx))


@RULES.of(Frac)
def _frac(p: LatexPrinter, e: Frac, ctx: Ctx) -> Res:
    return _delimited(r'\operatorname{frac}{\left(%s\right)}' % p.inside(e.args[0], ctx))


@RULES.of(Conjugate)
def _conjugate(p: LatexPrinter, e: Conjugate, ctx: Ctx) -> Res:
    return _delimited(r'\overline{%s}' % p.inside(e.args[0], ctx))


@RULES.of(Re)
def _re(p: LatexPrinter, e: Re, ctx: Ctx) -> Res:
    head = r'\Re' if p.settings.gothic_re_im else r'\operatorname{re}'
    return _applied(r'%s{\left(%s\right)}' % (head, p.inside(e.args[0], ctx)))


@RULES.of(Im)
def _im(p: LatexPrinter, e: Im, ctx: Ctx) -> Res:
    head = r'\Im' if p.settings.gothic_re_im else r'\operatorname{im}'
    return _applied(r'%s{\left(%s\right)}' % (head, p.inside(e.args[0], ctx)))


@RULES.of(Adjoint)
def _adjoint(p: LatexPrinter, e: Adjoint, ctx: Ctx) -> Res:
    # the mark is a script of the operand (SymPy: A^{a}^{\dagger}, \int A\, dx^{\dagger})
    return Result(r'%s^{\dagger}' % p.print(e.args[0], ctx.base(None)).output, Binding.POWER)


@RULES.of(Transpose)
def _transpose(p: LatexPrinter, e: Transpose, ctx: Ctx) -> Res:
    return Result('%s^{T}' % p.print(e.args[0], ctx.base(None)).output, Binding.POWER)


@RULES.of(Max, Min)
def _extremum(p: LatexPrinter, e: Application, ctx: Ctx) -> Res:
    args = sorted(e.args, key=default_sort_key)
    return _delimited(r'\%s\left(%s\right)' % (e.name.lower(), p.join(args, ctx)))


_INVERSE: tuple[type[Application], ...] = (Asin, Acos, Atan, Acsc, Asec, Acot, Asinh, Acosh, Atanh, Acsch, Asech, Acoth)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_Function (the inverse trigonometric styles)
def _inverse(p: LatexPrinter, e: Application, ctx: Ctx) -> Res:
    name = e.name
    style = p.settings.inv_trig_style
    if style == 'full':
        name = ('ar' if name.endswith('h') else 'arc') + name[1:]
    elif style == 'power':
        name = name[1:]
        head = ('\\%s' if name in ACCEPTED_FUNCTIONS else r'\operatorname{%s}') % name + '^{-1}'
        # the head has its superscript: an exponent goes around the whole
        return p.apply(head, e.args, ctx.without_offers())
    head = '\\' + name if name in ACCEPTED_FUNCTIONS else r'\operatorname{%s}' % name
    return p.apply(head, e.args, ctx)


_register(_INVERSE, _inverse)


# ----------------------------------------------------------------------
# factorials


@RULES.of(Factorial)
def _factorial(p: LatexPrinter, e: Factorial, ctx: Ctx) -> Res:
    return _applied('%s!' % p.print(e.args[0], ctx.operand()).output)


@RULES.of(Factorial2)
def _factorial2(p: LatexPrinter, e: Factorial2, ctx: Ctx) -> Res:
    return _applied('%s!!' % p.print(e.args[0], ctx.operand()).output)


@RULES.of(Subfactorial)
def _subfactorial(p: LatexPrinter, e: Subfactorial, ctx: Ctx) -> Res:
    return _applied('!%s' % p.print(e.args[0], ctx.operand()).output)


@RULES.of(Binomial)
def _binomial(p: LatexPrinter, e: Binomial, ctx: Ctx) -> Res:
    return _delimited(r'{\binom{%s}{%s}}' % (p.inside(e.args[0], ctx), p.inside(e.args[1], ctx)))


@RULES.of(RisingFactorial)
def _rising_factorial(p: LatexPrinter, e: RisingFactorial, ctx: Ctx) -> Res:
    base = p.print(e.args[0], ctx.operand()).output
    return Result(r'{%s}^{\left(%s\right)}' % (base, p.inside(e.args[1], ctx)), Binding.POWER)


@RULES.of(FallingFactorial)
def _falling_factorial(p: LatexPrinter, e: FallingFactorial, ctx: Ctx) -> Res:
    sub = p.print(e.args[1], ctx.subscript()).output
    return Result(r'{\left(%s\right)}_{%s}' % (p.inside(e.args[0], ctx), sub), Binding.ATOM, scripted=True)


@RULES.of(QPochhammer)
def _qpochhammer(p: LatexPrinter, e: QPochhammer, ctx: Ctx) -> Res:
    a, q, n = e.args
    tex = r'\left(%s; %s\right)_{%s}' % (p.inside(a, ctx), p.inside(q, ctx), p.print(n, ctx.subscript()).output)
    return Result(tex, Binding.ATOM, scripted=True)


# ----------------------------------------------------------------------
# the functions written as a letter and its arguments

#: the heads of the functions printed as ``head\left(arguments\right)``
_LETTERS: tuple[tuple[type[Application], str], ...] = (
    (Gamma, r'\Gamma'), (LowerGamma, r'\gamma'), (UpperGamma, r'\Gamma'), (Beta, r'\operatorname{B}'),
    (Zeta, r'\zeta'), (DirichletEta, r'\eta'), (LerchPhi, r'\Phi'), (ChiNode, r'\operatorname{Chi}'),
    (FresnelS, 'S'), (FresnelC, 'C'), (AiryAi, 'Ai'), (AiryBi, 'Bi'), (Mobius, r'\mu'),
    (MathieuC, 'C'), (MathieuS, 'S'),
)

_LETTER_OF: dict[type[Application], str] = dict(_LETTERS)


def _letter(p: LatexPrinter, e: Application, ctx: Ctx) -> Res:
    return p.attach(_LETTER_OF[type(e)], _round(p, e.args, ctx), ctx)


_register([cls for cls, _ in _LETTERS], _letter)

#: the heads with a superscript of their own, which take no exponent
_MARKED: tuple[tuple[type[Application], str], ...] = (
    (AiryAiPrime, r'Ai^\prime'), (AiryBiPrime, r'Bi^\prime'), (MathieuCPrime, r'C^{\prime}'),
    (MathieuSPrime, r'S^{\prime}'),
)

_MARKED_OF: dict[type[Application], str] = dict(_MARKED)


def _marked(p: LatexPrinter, e: Application, ctx: Ctx) -> Res:
    return p.attach(_MARKED_OF[type(e)], _round(p, e.args, ctx), ctx, scripted_head=True)


_register([cls for cls, _ in _MARKED], _marked)

#: the functions which SymPy brackets whole under an exponent
_WHOLE: tuple[tuple[type[Application], str], ...] = (
    (Totient, r'\phi'), (ReducedTotient, r'\lambda'), (PrimeNu, r'\nu'), (PrimeOmega, r'\Omega'),
)

_WHOLE_OF: dict[type[Application], str] = dict(_WHOLE)


def _whole(p: LatexPrinter, e: Application, ctx: Ctx) -> Res:
    return _applied(_WHOLE_OF[type(e)] + _round(p, e.args, ctx))


_register([cls for cls, _ in _WHOLE], _whole)


def _bar(p: LatexPrinter, before: Sequence[Expr], after: Expr, ctx: Ctx) -> str:
    r"""``\left(a; b\middle| m\right)``."""
    return r'\left(%s\middle| %s\right)' % ('; '.join(p.inside(a, ctx) for a in before), p.inside(after, ctx))


@RULES.of(EllipticK)
def _elliptic_k(p: LatexPrinter, e: EllipticK, ctx: Ctx) -> Res:
    return p.attach('K', _round(p, e.args, ctx), ctx)


@RULES.of(EllipticF)
def _elliptic_f(p: LatexPrinter, e: EllipticF, ctx: Ctx) -> Res:
    return p.attach('F', _bar(p, e.args[:1], e.args[1], ctx), ctx)


@RULES.of(EllipticE)
def _elliptic_e(p: LatexPrinter, e: EllipticE, ctx: Ctx) -> Res:
    if len(e.args) == 2:
        return p.attach('E', _bar(p, e.args[:1], e.args[1], ctx), ctx)
    return p.attach('E', _round(p, e.args, ctx), ctx)


@RULES.of(EllipticPi)
def _elliptic_pi(p: LatexPrinter, e: EllipticPi, ctx: Ctx) -> Res:
    return p.attach(r'\Pi', _bar(p, e.args[:-1], e.args[-1], ctx), ctx)


@RULES.of(BetaInc)
def _betainc(p: LatexPrinter, e: BetaInc, ctx: Ctx) -> Res:
    return _incomplete_beta(p, 'B', e.args, ctx)


@RULES.of(BetaIncRegularized)
def _betainc_regularized(p: LatexPrinter, e: BetaIncRegularized, ctx: Ctx) -> Res:
    return _incomplete_beta(p, 'I', e.args, ctx)


def _incomplete_beta(p: LatexPrinter, letter: str, args: Sequence[Expr], ctx: Ctx) -> Res:
    below = ctx.subscript()
    bounds = '(%s, %s)' % (p.print(args[2], below).output, p.print(args[3], below).output)
    return p.attach(r'\operatorname{%s}_{%s}' % (letter, bounds), _round(p, args[:2], ctx), ctx)


@RULES.of(Expint)
def _expint(p: LatexPrinter, e: Expint, ctx: Ctx) -> Res:
    nu = p.print(e.args[0], ctx.subscript()).output
    return p.attach(r'\operatorname{E}_{%s}' % nu, _round(p, e.args[1:], ctx), ctx)


@RULES.of(PolyLog)
def _polylog(p: LatexPrinter, e: PolyLog, ctx: Ctx) -> Res:
    s = p.print(e.args[0], ctx.subscript()).output
    return p.attach(r'\operatorname{Li}_{%s}' % s, _round(p, e.args[1:], ctx), ctx)


@RULES.of(Stieltjes)
def _stieltjes(p: LatexPrinter, e: Stieltjes, ctx: Ctx) -> Res:
    tail = '_{%s}' % p.print(e.args[0], ctx.subscript()).output
    if len(e.args) == 2:
        tail += _round(p, e.args[1:], ctx)
    return p.attach(r'\gamma', tail, ctx)


@RULES.of(LambertW)
def _lambertw(p: LatexPrinter, e: LambertW, ctx: Ctx) -> Res:
    tail = _round(p, e.args[:1], ctx)
    if len(e.args) == 2:
        tail = '_{%s}%s' % (p.print(e.args[1], ctx.subscript()).output, tail)
    return p.attach('W', tail, ctx)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_divisor_sigma
@RULES.of(DivisorSigma)
def _divisor_sigma(p: LatexPrinter, e: DivisorSigma, ctx: Ctx) -> Res:
    return p.attach(r'\sigma', _sigma_tail(p, e.args, ctx), ctx)


@RULES.of(UDivisorSigma)
def _udivisor_sigma(p: LatexPrinter, e: UDivisorSigma, ctx: Ctx) -> Res:
    # the head has a superscript: an exponent goes around the whole (SymPy: \sigma^*^{2})
    return p.attach(r'\sigma^*', _sigma_tail(p, e.args, ctx), ctx, scripted_head=True)


def _sigma_tail(p: LatexPrinter, args: Sequence[Expr], ctx: Ctx) -> str:
    tail = _round(p, args[:1], ctx)
    if len(args) == 2:
        # the subscript in braces (SymPy: \sigma_k + 1, \sigma_10)
        tail = '_{%s}%s' % (p.print(args[1], ctx.subscript()).output, tail)
    return tail


# ----------------------------------------------------------------------
# Bessel functions, orthogonal polynomials, number sequences

_BESSEL: tuple[tuple[type[Application], str, bool], ...] = (
    (BesselJ, 'J', False), (BesselI, 'I', False), (BesselK, 'K', False), (BesselY, 'Y', False),
    (SphericalJ, 'j', False), (SphericalY, 'y', False), (Hankel1, 'H^{(1)}', True), (Hankel2, 'H^{(2)}', True),
    (SphericalHankel1, 'h^{(1)}', True), (SphericalHankel2, 'h^{(2)}', True),
)

_BESSEL_OF: dict[type[Application], tuple[str, bool]] = {cls: (head, marked) for cls, head, marked in _BESSEL}


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._hprint_BesselBase
def _bessel(p: LatexPrinter, e: Application, ctx: Ctx) -> Res:
    head, marked = _BESSEL_OF[type(e)]
    order = p.print(e.args[0], ctx.subscript()).output
    return p.attach(head, '_{%s}%s' % (order, _round(p, e.args[1:], ctx)), ctx, scripted_head=marked)


_register([cls for cls, _, _ in _BESSEL], _bessel)

#: the letter, the number of parameters in the superscript (in round brackets)
_POLYNOMIALS: tuple[tuple[type[Application], str, int], ...] = (
    (Jacobi, 'P', 2), (Gegenbauer, 'C', 1), (ChebyshevT, 'T', 0), (ChebyshevU, 'U', 0), (Legendre, 'P', 0),
    (AssocLegendre, 'P', 1), (Hermite, 'H', 0), (Laguerre, 'L', 0), (AssocLaguerre, 'L', 1),
)

_POLYNOMIAL_OF: dict[type[Application], tuple[str, int]] = {cls: (letter, n) for cls, letter, n in _POLYNOMIALS}


def _polynomial(p: LatexPrinter, e: Application, ctx: Ctx) -> Res:
    letter, parameters = _POLYNOMIAL_OF[type(e)]
    tex = '%s_{%s}' % (letter, p.print(e.args[0], ctx.subscript()).output)
    if parameters:
        above = ctx.superscript()
        tex += r'^{\left(%s\right)}' % ','.join(p.print(a, above).output for a in e.args[1:1 + parameters])
    return _applied(tex + _round(p, e.args[1 + parameters:], ctx))


_register([cls for cls, _, _ in _POLYNOMIALS], _polynomial)


@RULES.of(YnmNode, ZnmNode)
def _spherical_harmonic(p: LatexPrinter, e: Application, ctx: Ctx) -> Res:
    n, m, theta, phi = e.args
    letter = 'Y' if isinstance(e, YnmNode) else 'Z'
    tex = '%s_{%s}^{%s}' % (letter, p.print(n, ctx.subscript()).output, p.print(m, ctx.superscript()).output)
    return _applied(r'%s\left(%s,%s\right)' % (tex, p.inside(theta, ctx), p.inside(phi, ctx)))


_NUMBERS: tuple[tuple[type[Application], str], ...] = (
    (Bernoulli, 'B'), (Genocchi, 'G'), (Bell, 'B'), (Fibonacci, 'F'), (Lucas, 'L'), (Tribonacci, 'T'),
    (Euler, 'E'), (Catalan, 'C'),
)

_NUMBER_OF: dict[type[Application], str] = dict(_NUMBERS)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter.__print_number_polynomial
def _number(p: LatexPrinter, e: Application, ctx: Ctx) -> Res:
    tail = '_{%s}' % p.print(e.args[0], ctx.subscript()).output
    if len(e.args) == 1:
        # an indexed letter: an exponent stands after the index
        return Result(_NUMBER_OF[type(e)] + tail, Binding.ATOM)
    return p.attach(_NUMBER_OF[type(e)], tail + _round(p, e.args[1:], ctx), ctx)


_register([cls for cls, _ in _NUMBERS], _number)


# ----------------------------------------------------------------------
# hypergeometric functions


def _vector(p: LatexPrinter, parameters: Sequence[Expr], ctx: Ctx) -> str:
    return p.join(parameters, ctx)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_hyper
@RULES.of(Hyper)
def _hyper(p: LatexPrinter, e: Hyper, ctx: Ctx) -> Res:
    tex = r'{{}_{%d}F_{%d}\left(\begin{matrix} %s \\ %s \end{matrix}\middle| {%s} \right)}' % (
        len(e.ap), len(e.bq), _vector(p, e.ap, ctx), _vector(p, e.bq, ctx), p.inside(e.argument, ctx))
    return _delimited(tex)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_meijerg
@RULES.of(MeijerG)
def _meijerg(p: LatexPrinter, e: MeijerG, ctx: Ctx) -> Res:
    tex = r'{G_{%d, %d}^{%d, %d}\left(\begin{matrix} %s & %s \\%s & %s \end{matrix} \middle| {%s} \right)}' % (
        len(e.ap), len(e.bq), len(e.bm), len(e.an), _vector(p, e.an, ctx), _vector(p, e.aother, ctx),
        _vector(p, e.bm, ctx), _vector(p, e.bother, ctx), p.inside(e.argument, ctx))
    return _delimited(tex)


# ----------------------------------------------------------------------
# delta functions and tensors


@RULES.of(DiracDelta)
def _dirac_delta(p: LatexPrinter, e: DiracDelta, ctx: Ctx) -> Res:
    if len(e.args) == 1 or e.args[1].is_zero is True:
        return _applied(r'\delta\left(%s\right)' % p.inside(e.args[0], ctx))
    order = p.print(e.args[1], ctx.superscript()).output
    return _applied(r'\delta^{\left( %s \right)}\left( %s \right)' % (order, p.inside(e.args[0], ctx)))


@RULES.of(Heaviside)
def _heaviside(p: LatexPrinter, e: Heaviside, ctx: Ctx) -> Res:
    return _applied(r'\theta\left(%s\right)' % p.join(e.pargs, ctx))


@RULES.of(KroneckerDeltaNode)
def _kronecker_delta(p: LatexPrinter, e: KroneckerDeltaNode, ctx: Ctx) -> Res:
    below = ctx.subscript()
    separator = ' ' if all(a.is_Atom for a in e.args[:2]) else ', '
    return Result(r'\delta_{%s}' % separator.join(p.print(a, below).output for a in e.args[:2]), Binding.ATOM)


@RULES.of(LeviCivitaNode)
def _levi_civita(p: LatexPrinter, e: LeviCivitaNode, ctx: Ctx) -> Res:
    below = ctx.subscript()
    separator = ' ' if all(a.is_Atom for a in e.args) else ', '
    return Result(r'\varepsilon_{%s}' % separator.join(p.print(a, below).output for a in e.args), Binding.ATOM)


# PORTED FROM SYMPY: sympy/printing/latex.py:LatexPrinter._print_SingularityFunction
@RULES.of(SingularityFunctionNode)
def _singularity_function(p: LatexPrinter, e: SingularityFunctionNode, ctx: Ctx) -> Res:
    x, a, n = e.args
    shift = p.inside(add(x, neg(a)), ctx)
    power = p.print(n, ctx.superscript()).output
    return Result(r'{\left\langle %s \right\rangle}^{%s}' % (shift, power), Binding.POWER)
