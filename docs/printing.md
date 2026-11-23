# Printing: LaTeX

`sympy_extras.printing` prints the native objects (the nodes of the typed
core, `ExprMatrix`, `Poly`, Python containers of them):

* `latex(expr, **settings)`: the LaTeX form, with the settings of SymPy's
  `latex`;
* `init_printing(...)`: how the results are shown in a notebook or in a
  Python session; every native object has a `_repr_latex_` method, which
  Jupyter renders;
* `sympy_extras.printing.framework`: the typed framework the printers
  share (it does not depend on LaTeX: the pretty printer uses it with
  pictures instead of strings).

```python
>>> from sympy_extras.core import symbols, Rational, Function, Derivative, Eq
>>> from sympy_extras.functions.elementary.trigonometric import sin, cos
>>> from sympy_extras.functions.elementary.exponential import exp
>>> from sympy_extras.printing import latex
>>> x, y, n = symbols('x y n')
>>> f = Function('f')
>>> print(latex(sin(x)**2 + exp(-x)/(2*y)))
\sin^{2}{\left(x \right)} + \frac{e^{- x}}{2 y}
>>> print(latex(Eq(Derivative(f(x), x), x**Rational(3, 2))))
\frac{d}{d x} f{\left(x \right)} = x^{\frac{3}{2}}

```

Sums, products, integrals, limits, piecewise expressions, sets, matrices
and polynomials:

```python
>>> from sympy_extras.core import Integer, oo
>>> from sympy_extras.core.limits import Limit
>>> from sympy_extras.concrete.expr_with_limits import Sum
>>> from sympy_extras.integrals.integral import Integral
>>> from sympy_extras.functions.elementary.piecewise import Piecewise
>>> from sympy_extras.sets.sets import Interval, Union, FiniteSet
>>> from sympy_extras.matrices.dense import ExprMatrix
>>> from sympy_extras.polys.polytools import Poly
>>> k = symbols('k')
>>> print(latex(Sum(1/k**2, (k, 1, oo))))
\sum_{k=1}^{\infty} \frac{1}{k^{2}}
>>> print(latex(Integral(exp(-x**2), x, Integer(0), oo)))
\int\limits_{0}^{\infty} e^{- x^{2}}\, dx
>>> print(latex(Limit(sin(x)/x, x, 0)))
\lim_{x \to 0^+} \frac{\sin{\left(x \right)}}{x}
>>> print(latex(Piecewise((x, x < 0), (x**2, True))))
\begin{cases} x & \text{for}\: x < 0 \\x^{2} & \text{otherwise} \end{cases}
>>> print(latex(Union(Interval(0, 1), FiniteSet(2, 3))))
\left[0, 1\right] \cup \left\{2, 3\right\}
>>> print(latex(ExprMatrix.from_rows([[x, 1], [0, cos(y)]])))
\left[\begin{matrix}x & 1\\0 & \cos{\left(y \right)}\end{matrix}\right]
>>> print(latex(Poly(x**2 - 2*x + 1, x)))
\operatorname{Poly}{\left( x^{2} - 2 x + 1, x, domain=\mathbb{Z} \right)}

```

## Settings

The settings are keyword arguments, typed; a wrong value raises
`ValueError`.

| setting | meaning | default |
|---|---|---|
| `mode` | `'plain'`, `'inline'` (`$...$`, short fractions on one line), `'equation'`, `'equation*'` | `'plain'` |
| `fold_frac_powers` | `x^{2/3}` instead of `x^{\frac{2}{3}}` | `False` |
| `fold_func_brackets` | `\sin {x}` where the argument can stand bare | `False` |
| `fold_short_frac` | `x / 2` when the denominator is short | `True` in the inline mode |
| `inv_trig_style` | `'abbreviated'` (`asin`), `'full'` (`\arcsin`), `'power'` (`\sin^{-1}`) | `'abbreviated'` |
| `itex` | the itex dialect (`$$...$$`, `\root`, `\gt`) | `False` |
| `ln_notation` | `\ln` instead of `\log` | `False` |
| `long_frac_ratio` | above this ratio of the widths, a part of the numerator goes after the fraction | `None` |
| `mat_delim`, `mat_str` | the delimiter (`'['`, `'('`, `''`) and the environment of matrices | `'['`, by size |
| `mul_symbol` | `None`, `'ldot'`, `'dot'`, `'times'` or any LaTeX | `None` |
| `order` | the order of the terms of sums: `None`/`'lex'`, `'grlex'`, `'grevlex'`, `'none'` (as stored) | `None` |
| `symbol_names` | the LaTeX of some symbols | `{}` |
| `root_notation` | `\sqrt[3]{x}` instead of `x^{\frac{1}{3}}` | `True` |
| `imaginary_unit` | `'i'`, `'ri'`, `'ti'`, `'j'`, `'rj'`, `'tj'` or any LaTeX | `'i'` |
| `gothic_re_im` | `\Re`, `\Im` | `False` |
| `decimal_separator` | `'period'` or `'comma'` (`2{,}5`; elements and arguments then separated by `;`) | `'period'` |
| `parenthesize_super` | `\left(x^{i}\right)^{2}` or `{x^{i}}^{2}` | `True` |
| `min`, `max` | the exponents beyond which a float is printed in scientific notation | `None` |
| `full_prec` | all the digits of a float | `False` |
| `diff_operator` | `'d'`, `'rd'`, `'td'` or any LaTeX | `'d'` |

```python
>>> print(latex(x**Rational(2, 3)/2, fold_frac_powers=True, mode='inline'))
$x^{2/3} / 2$
>>> print(latex(2*x*y, mul_symbol='dot'), latex(sin(x), fold_func_brackets=True))
2 \cdot x \cdot y \sin {x}
>>> from sympy_extras.functions.elementary.trigonometric import asin
>>> print(latex(asin(x), inv_trig_style='full'), latex(asin(x), inv_trig_style='power'))
\arcsin{\left(x \right)} \sin^{-1}{\left(x \right)}
>>> print(latex(x, symbol_names={x: r'\xi'}), latex(1.5*x, decimal_separator='comma'))
\xi 1{,}5 x

```

SymPy's `mat_symbol_style`, `perm_cyclic`, `adjoint_style` and
`disable_split_super_sub` concern classes this package does not have (or
a way to switch off the scripts of names which produced invalid TeX) and
are not offered; the orders `'old'` and the reversed ones are not
supported.

## Design

SymPy's printers find the method of a node by the *name* of its class
(`_print_<ClassName>`, with `getattr` along the bases), and each parent
decides on brackets from a precedence number computed from the *class* of
the child, whatever the child printed. Most of the defects listed in
[sympy/sympy#30844](https://github.com/sympy/sympy/issues/30844) come from
there: a negative product has the precedence of a product, the sides of a
relation are never bracketed, `Not` tests `is_Boolean`, the exponent of a
power is passed into the printer of its base as an `exp=` keyword which
half of the printers do not take. The printers of this package are built
differently (`sympy_extras/printing/framework.py`):

1. **Dispatch by class.** The rules are typed functions
   `rule(printer, node, context)` in a table from classes to rules
   (`Dispatcher`); the rule of a node is the one of its class or of the
   nearest base which has one. No name is ever looked up: in a statically
   typed language the table is a `match` on the node. The rules live in
   the printing package, not on the nodes: a printer is one module, the
   core does not grow a method per printer, and the objects which are not
   nodes (matrices, polynomials, Python containers) are printed the same
   way. `test_latex_coverage.py` enumerates the subclasses of `Basic` and
   checks that every one finds a rule.
2. **A rule returns what it printed and how it binds.** A `Result` is the
   output with the `Binding` of its outermost operator *as printed*: `- x`
   and `-2` bind as a negation whatever their class, `\frac{1}{x}` (the
   power `x**-1`) as a product, `x / 2` as a quotient,
   `\sin^{2}{\left(x \right)}` as a power, an `AlgebraicNumber` or an
   `UnevaluatedExpr` as what its content printed. A few flags complete
   it: the output without its leading minus sign (a sum writes `a - b`),
   an open-ended construct on the right (a sum, a limit, a derivative, an
   integral), a script at the end.
3. **A parent passes a context down.** A `Context` is immutable and says
   where the child stands and what the position requires: a term, a
   factor (followed by another or not), a numerator or a denominator
   (built up or after a slash), a base with the exponent it may render
   itself, an exponent, a subscript, the operand of `!`, a side of a
   relation, an operand of a connective, the operand of a negation, the
   body of a sum or an integral, the point of a limit. It also carries
   the inherited state (inside an exponent, inline). The settings are
   never modified while printing.
4. **Brackets are decided in one place**, `needs_brackets(result,
   context)`, called by `TreePrinter.print` for every child. No rule adds
   brackets of its own. What depends on the parent is in the context:
   `sin(x)**2` is `\sin^{2}{\left(x \right)}` because the power *offers*
   its printed exponent to its base and the rule of the function takes
   it (the result is then marked as standing for the power);
   `Not(Implies(a, b))` is `a \not\Rightarrow b` the same way; the mark of
   the direction of a limit is a script of the point, so that the point
   is printed in the position of a base.

```python
>>> from sympy_extras.printing.framework import Binding, Context, Result, needs_brackets
>>> top: Context[str] = Context()
>>> minus_x = Result('- x', Binding.NEGATION, unsigned='x')
>>> [needs_brackets(minus_x, c) for c in (top.term(), top.factor(False), top.base(None), top.exponent())]
[False, True, True, False]
>>> from sympy_extras.core import Mul, Pow, Not, Mod
>>> print(latex(Mul(2, Mul(-1, y))), latex(Pow(Rational(-1, 2), x)), latex(Not(x < y)), latex(Mod(-x, y)))
2 \left(- y\right) \left(- \frac{1}{2}\right)^{x} \neg \left(x < y\right) \left(- x\right) \bmod y

```

### A rule for a new class

A rule is registered for a class in the table of the printer; it prints
its children through the printer, each in the context of its position:

```python
>>> from sympy_extras.core.function import DefinedFunction
>>> from sympy_extras.printing.framework import Binding, Result
>>> from sympy_extras.printing.latex import RULES, Ctx, LatexPrinter, Res
>>> class Norm(DefinedFunction):
...     function_name = 'norm'
>>> @RULES.of(Norm)
... def _norm(printer: LatexPrinter, e: Norm, context: Ctx) -> Res:
...     return Result(r'\left\|%s\right\|' % printer.inside(e.args[0], context), Binding.ATOM)
>>> print(latex(Norm(x + y)**2/2))
\frac{\left\|x + y\right\|^{2}}{2}

```

## Tests

* `test_latex_sympy.py`: a corpus of about 390 expressions prints exactly
  as in SymPy under 14 combinations of settings;
* `test_latex_semantic.py`: 2100 seeded random unevaluated trees (negative
  coefficients, nested powers and quotients, functions, `mod`, relations,
  connectives) are printed under five settings, read back by a small
  LaTeX reader written for the tests (`tests/latex_reader.py`) and
  evaluated at random points: the value must be the value of the tree.
  About 1800 formulas (the corpus and random trees, under three settings)
  are compiled with pdflatex when it is installed;
* `test_latex_issue_30844.py`: one test per item of the issue which
  concerns a class of this package;
* `test_latex_coverage.py`: every node class has a rule, every function
  prints alone, as a factor and under an exponent.

## Differences from SymPy

Besides the defects of the issue, the output differs from SymPy's where
SymPy's is ambiguous or inconsistent:

| expression | SymPy | here |
|---|---|---|
| `x/(2*y)` on one line (`fold_short_frac`) | `x / 2 y` | `x / \left(2 y\right)` |
| `factorial(n)**2` | `n!^{2}` | `\left(n!\right)^{2}` |
| `Limit(sin(x)/x, x, 0)` | the body always bracketed | bracketed when it is a sum |
| `Piecewise(...) + 1` | the cases bare | the cases bracketed |
| `asin(x)**2` with `inv_trig_style='power'` | `\sin^{-1}{\left(x \right)}^{2}` | `\left(\sin^{-1}{\left(x \right)}\right)^{2}` |
| `airyaiprime(x)**2` | `{Ai^\prime}^{2}\left(x\right)` | `\left(Ai^\prime\left(x\right)\right)^{2}` |
| `Mul(-1, x + y)` (unevaluated) | `- (x + y)` | `- \left(x + y\right)` |
| `Mul(Mul(-1, x), y)` (unevaluated) | `- x y` | `\left(- x\right) y` |
| `Pow(Float(2.5), 2)` | `{2.5}^{2}` | `2.5^{2}` |
| `1/sqrt(x)` on one line | `\frac{1}{\sqrt{x}}` | `1 / \sqrt{x}` |
| `Poly(x**2*y - x, x, y)` | `x^{2}y -x` | `x^{2} y - x` |
| `decimal_separator='comma'` | `;` in lists, tuples and sets only | `;` between all elements and arguments |
| `ExpPolar(x)` | `e^{x}` | `\operatorname{exp\_polar}{\left(x \right)}` |
| `4**(-2*x)*7**(2*x)`, `Mul(x, 3)` | `4^{- 2 x} 7^{2 x}`, `x 3` | `4^{- 2 x} \cdot 7^{2 x}`, `x \cdot 3`: a number is never juxtaposed to what precedes it |
| `exp(Rational(-1, 2))` with `fold_frac_powers` | `e^{- \frac{1}{2}}` | `e^{-1/2}`: every rational exponent is folded |
| a sum with a `Mod` somewhere inside a term | the whole term bracketed | only the `mod` itself |
| `Interval(0, oo)` built closed | — | printed open at the infinite end |

## Interactive sessions

```python
>>> from sympy_extras.printing import init_printing
>>> from sympy_extras.printing.interactive import reset_printing
>>> print((x**2/2)._repr_latex_())
$\displaystyle \frac{x^{2}}{2}$
>>> init_printing(mul_symbol='dot', display_hook=False)
>>> print((2*x)._repr_latex_())
$\displaystyle 2 \cdot x$
>>> reset_printing()

```

`init_printing(use_latex=False)` makes `_repr_latex_` return `None`, so
that a notebook shows the text. In a plain Python session
`init_printing()` installs a display hook which shows the native objects
with the text printer: the string form, or the two-dimensional printer
plugged in with `sympy_extras.printing.interactive.set_pretty_printer`
(a function of the object and of whether Unicode may be used), which is
the hook the pretty printer of this package uses. Nothing of IPython is
imported or configured.
