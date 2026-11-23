# Printing: LaTeX and pretty printing

`sympy_extras.printing` prints the native objects (the nodes of the typed
core, `ExprMatrix`, `Poly`, Python containers of them) without importing
SymPy:

* `latex(expr, **settings)`: the LaTeX form, with the settings of SymPy's
  `latex`;
* `pretty(expr, **settings)` and `pprint`: two-dimensional text, in
  Unicode or in ASCII (`sympy_extras.printing.pretty`): fraction bars,
  raised exponents, root signs, sum, product and integral signs, braces of
  cases;
* `init_printing(...)`: how the results are shown in a notebook or in a
  Python session; every native object has a `_repr_latex_` method, which
  Jupyter renders, and a `_repr_pretty_` method, which IPython calls for
  the text;
* `sympy_extras.printing.framework`: the typed framework the two printers
  share (rules found by class, typed results and contexts, brackets in one
  place), and `sympy_extras.printing.conventions`, the reading of names
  they share.

Both printers are typed ports of SymPy's (`sympy/printing/latex.py`,
`sympy/printing/pretty`), which they follow character by character where
SymPy is right, on another design of the dispatch and of the bracketing.
The defects of SymPy's printers listed in
[sympy/sympy#30844](https://github.com/sympy/sympy/issues/30844) which
concern the classes of this package are not reproduced.

## LaTeX

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

### Settings of `latex`

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

## Pretty printing

```python
>>> from sympy_extras.core import symbols, Rational, sqrt, oo, Pow, Not
>>> from sympy_extras.functions import sin, exp
>>> from sympy_extras.integrals.integral import Integral
>>> from sympy_extras.concrete.expr_with_limits import Sum
>>> from sympy_extras.printing.pretty import pretty, pprint
>>> x, y, n, k = symbols('x y n k')
>>> pprint((x**2 + 1)/(y - 1))  # doctest: +NORMALIZE_WHITESPACE
 2
x  + 1
------
y - 1
>>> pprint(Integral(exp(-x**2), x, -oo, oo), use_unicode=True)  # doctest: +NORMALIZE_WHITESPACE
∞
⌠
⎮     2
⎮   -x
⎮  ℯ    dx
⌡
-∞
>>> pprint(Sum(1/k**2, (k, 1, n)) + sqrt(x), use_unicode=True)  # doctest: +NORMALIZE_WHITESPACE
      n
     ____
     ╲
      ╲    1
       ╲   ──
√x +   ╱    2
      ╱    k
     ╱
     ‾‾‾‾
     k = 1

```

### Functions and settings of `pretty`

| Function | |
|---|---|
| `pretty(expr, use_unicode=None, wrap_line=True, num_columns=None, order=None, root_notation=True, use_unicode_sqrt_char=True, imaginary_unit='i', full_prec=None)` | the picture as a string; its rows have the same width |
| `pprint(expr, ...)`, `pretty_print` | prints it |
| `pretty_picture(expr, settings)` | the `Picture` (rows and baseline), not cut into screens |
| `pretty_use_unicode(flag=None)` | the default of `use_unicode` (ASCII at the start); returns the previous value |
| `pretty_try_use_unicode()` | makes Unicode the default when the standard output can encode it |
| `pretty_repr(expr, sink, cycle, settings=None)` | the pretty form written to the printer which IPython hands to a `_repr_pretty_(self, p, cycle)` method (the native objects use `interactive.repr_pretty`, which follows `init_printing`) |

`expr` is a node, an `ExprMatrix`, a `Poly`, or a Python number, string,
list, tuple, dictionary or set of such objects.

| Setting | |
|---|---|
| `use_unicode` | Unicode or ASCII; `None` is the default set by `pretty_use_unicode` |
| `wrap_line`, `num_columns` | a picture wider than `num_columns` (the terminal for `None`) is cut into screens, the cuts marked with `↪` or `>` |
| `order` | `None` (or `'lex'`): the printing order of `sstr`; `'none'`: the order of the arguments |
| `root_notation` | `False` writes `x**(1/n)` as a power, not as a root |
| `use_unicode_sqrt_char` | `False` draws a square root always with its bar, never as `√x` |
| `imaginary_unit` | `'i'` or `'j'`: the letter of the imaginary unit in Unicode |
| `full_prec` | `True`: the floats with all the digits of their precision; `False`: without the trailing zeros; `None` (SymPy's `'auto'`): all the digits for a float which is the whole expression |

```python
>>> print(pretty(x**Rational(1, 3)), '|', pretty(sqrt(x), use_unicode=True))  # doctest: +NORMALIZE_WHITESPACE
3 ___
\/ x  | √x
>>> pprint(x**Rational(1, 3), root_notation=False)  # doctest: +NORMALIZE_WHITESPACE
 1/3
x
>>> pprint([x/2, {x: sin(x)**2}], use_unicode=True)  # doctest: +NORMALIZE_WHITESPACE
⎡x  ⎧      2   ⎫⎤
⎢─, ⎨x: sin (x)⎬⎥
⎣2  ⎩          ⎭⎦
>>> from sympy_extras.core import Float
>>> from sympy_extras.core.logic import ITE
>>> print(pretty(Float('0.5')), '|', pretty(Float('0.5')*x), '|', pretty(Float('0.5')*x, full_prec=True))
0.500000000000000 | 0.5*x | 0.500000000000000*x
>>> pprint(ITE(x > 0, y > 0, n > 0), use_unicode=True)  # doctest: +NORMALIZE_WHITESPACE
⎧y > 0  for x > 0
⎨
⎩n > 0  otherwise

```

## Design

SymPy's printers find the method of a node by the *name* of its class
(`_print_<ClassName>`, with `getattr` along the bases), and each parent
decides on brackets from a precedence number computed from the *class* of
the child, whatever the child printed. Most of the defects listed in
sympy/sympy#30844 come from there: a negative product has the precedence
of a product, the sides of a relation are never bracketed, `Not` tests
`is_Boolean`, the exponent of a power is passed into the printer of its
base as an `exp=` keyword which half of the printers do not take, a
`prettyForm` carries a binding but many rules build a composite picture
without setting it, so that it stays `ATOM`. The two printers of this
package are built differently, on one framework
(`sympy_extras/printing/framework.py`) which does not know the type of
the output: a string for `LatexPrinter`, a `Picture` (rows of text with a
baseline) for the pretty printer. Both are a `TreePrinter`.

1. **Dispatch by class.** The rules are typed functions
   `rule(printer, node, context)` in a table from classes to rules
   (`Dispatcher`; `latex.RULES`, `pretty.RULES`); the rule of a node is
   the one of its class or of the nearest base which has one. No name is
   ever looked up: in a statically typed language the table is a `match`
   on the node. The rules live in the printing package, not on the nodes:
   a printer is one module, the core does not grow a method per printer,
   and the objects which are not nodes (matrices, polynomials, Python
   containers) are printed the same way. `test_latex_coverage.py` and
   `test_pretty.py` enumerate the subclasses of `Basic` and check that
   every one finds a rule.
2. **A rule returns what it printed and how it binds.** A `Result` is the
   output with the `Binding` of its outermost operator *as printed* (no
   default: a rule must say it): `- x` and `-2` bind as a negation
   whatever their class, `\frac{1}{x}` (the power `x**-1`) as a product,
   `x / 2` as a quotient, `\sin^{2}{\left(x \right)}` and `sin²(x)` as a
   power, an `AlgebraicNumber` or an `UnevaluatedExpr` as what its content
   printed. A few flags complete it: `unsigned`, the output without its
   leading minus sign (a sum writes `a - b`); `open_ended`, a construct on
   the right whose body extends to the right (a sum, a limit, a
   derivative, an integral); `scripted`, a script at the end; `absorbed`,
   the rule rendered the operator its parent offered; `applies`, a
   function application outside any brackets (for `\sin {x}`); `stacked`,
   the outermost layout is a fraction bar drawn in two dimensions.
3. **A parent passes a context down.** A `Context` is immutable and says
   where the child stands (`Position`) and what the position requires: a
   term, a factor (followed by another or not), a numerator or a
   denominator (built up or after a slash), a base with the exponent it
   may render itself, an exponent, a subscript, the operand of `!`, a
   side of a relation, an operand of a connective, the operand of a
   negation, the body of a sum or an integral, the point of a limit. It
   also carries the inherited state (inside an exponent, inline). The
   methods of `Context` (`term()`, `factor(followed)`, `side()`, ...)
   make the positions whose requirements are the same in every notation;
   `Context.at(position, level, ...)` makes one whose requirements are
   those of the notation of one printer (the base of a power takes a
   function application in a picture, an atom or a script in LaTeX). The
   settings are never modified while printing.
4. **Brackets are decided in one place**, `needs_brackets(result,
   context)`, called by `TreePrinter.print` (and by `TreePrinter.enclose`,
   its second half, for the rules which look at the result of a child
   first) for every child. No rule adds brackets of its own; a printer
   only says how they are drawn (`\left( \right)`, parentheses as tall as
   the picture). What depends on the parent is in the context: `sin(x)**2`
   is `\sin^{2}{\left(x \right)}` and `sin²(x)` because the power *offers*
   its printed exponent to its base and the rule of the function takes it
   (the result is then marked as standing for the power);
   `Not(Implies(a, b))` is `a \not\Rightarrow b` and `a ↛ b` the same way;
   the mark of the direction of a limit is a script of the point, so that
   the point is printed in the position of a base. (The delimiters of a
   notation, `f(...)`, `|x|`, `d/dx(...)`, are not brackets of precedence.)

There is one scale of bindings for both printers, from the loosest to the
tightest. A printer says how **its own notation** binds, so that some
levels are produced by one printer only:

| `Binding` | printed forms | produced by |
|---|---|---|
| `LOWEST` | what a delimited position requires; `x ↦ y` and a brace of cases in a picture | both |
| `BLOCK` | `\begin{cases}`, a quantified formula | LaTeX |
| `IMPLICATION` | `⇒`, `⇔` (and `\veebar`) | both |
| `XOR` | `⊻` | pretty |
| `OR`, `AND` | the connectives | both |
| `RELATION` | `=`, `<`, `∈` | both |
| `MODULO` | `x mod y` | both |
| `SET_OPERATION` | `∪`, `∩`, `\`, `×` | both |
| `SUM` | sums | both |
| `NEGATION` | what starts with a minus sign: `-x`, `-2`, `-1/2`, `-2.5` | both |
| `QUOTIENT` | a fraction on one line: `1/2`, `x / 2` | both |
| `PRODUCT` | products, `\frac{a}{b}`, sum and product signs, integrals, limits, derivatives | both |
| `FRACTION` | a fraction with a bar drawn in a picture | pretty |
| `ROOT` | the one-character root sign `√x`, which no bar closes | pretty |
| `POWER` | powers, `\sin^{2}{\left(x \right)}`, `sin²(x)` | both |
| `AFFIX` | `¬x`, `x!`, `∀x ...` in a picture | pretty |
| `APPLICATION` | `f(x)`; `x!` in LaTeX | both |
| `NOT` | `\neg x` | LaTeX |
| `ATOM` | one token, or a form closed by its own delimiters | both |

```python
>>> from sympy_extras.printing.framework import Binding, Context, Result, needs_brackets
>>> top: Context[str] = Context()
>>> minus_x = Result('- x', Binding.NEGATION, unsigned='x')
>>> [needs_brackets(minus_x, c) for c in (top.term(), top.factor(False), top.base(None), top.exponent())]
[False, True, True, False]
>>> from sympy_extras.core import Mul, Pow, Not, Mod
>>> print(latex(Mul(2, Mul(-1, y))), latex(Pow(Rational(-1, 2), x)), latex(Not(x < y)), latex(Mod(-x, y)))
2 \left(- y\right) \left(- \frac{1}{2}\right)^{x} \neg \left(x < y\right) \left(- x\right) \bmod y
>>> print(pretty(Mul(2, Mul(-1, y))), '|', pretty(Not(x < y), use_unicode=True), '|', pretty(Mod(-x, y)))
2*(-y) | ¬(x < y) | (-x) mod y

```

The same expressions bind in the same way in both printers, except where
the notation differs (a quotient is a fraction bar in a picture and a
`\frac` in LaTeX):

```python
>>> from sympy_extras.printing.latex import LatexPrinter, LatexSettings
>>> from sympy_extras.printing.pretty import PrettyPrinter, PrettySettings, RULES as PRETTY_RULES, TOP
>>> pictures, strings = PrettyPrinter(PrettySettings(), PRETTY_RULES), LatexPrinter(LatexSettings())
>>> [pictures.rule(e, TOP).binding.name for e in (x, -x, x/y, Rational(1, 2), sin(x), sin(x)**2)]
['ATOM', 'NEGATION', 'FRACTION', 'QUOTIENT', 'APPLICATION', 'POWER']
>>> [strings.rule(e, Context()).binding.name for e in (x, -x, x/y, Rational(1, 2), sin(x), sin(x)**2)]
['ATOM', 'NEGATION', 'PRODUCT', 'PRODUCT', 'APPLICATION', 'POWER']

```

The names are read in one place too
(`sympy_extras/printing/conventions.py`: `split_super_sub`, the Greek
letters, the walk over the modifiers at the end of a name), and the terms
of the sums and the factors of the products are ordered by the helpers of
the core which `sstr` uses (`core.printing.printed_terms`,
`printed_factors`), so that the three printers write an expression in the
same order:

```python
>>> from sympy_extras.core import Symbol
>>> print(latex(Symbol('alphahat_1')), '|', latex(Symbol('lamda')), pretty(Symbol('lamda'), use_unicode=True))
\hat{\alpha}_{1} | \lambda λ
>>> e = exp(-2*x/3) + cos(y)
>>> print(e, '|', latex(e), '|', pretty(e, use_unicode=True).splitlines()[-1].rstrip())
cos(y) + exp(-2*x/3) | \cos{\left(y \right)} + e^{- \frac{2 x}{3}} | cos(y) + ℯ

```

What still differs between the two printers, and why:

* the output, and how the brackets are drawn (`TreePrinter.bracket`): a
  string between `\left(` and `\right)` (braces for a script on a
  script), a picture between parentheses of its height;
* the levels marked "LaTeX" or "pretty" in the table: LaTeX has built-up
  constructs which a picture draws with bars and tall signs, and the
  other way round;
* some flags are used by one printer only: `scripted`/`scripts_follow`,
  `applies` and `inline` by LaTeX (`{x^{i}}^{2}`, `\sin {x}`, the inline
  mode), `stacked`, `Context.excluded` and `Context.display` by the
  pretty printer (a fraction in a fraction, a power in a raised exponent,
  the fractions of a sum with tall terms);
* the base of a power: LaTeX takes an atom or anything with a script it
  can put in braces, a picture takes a function application (`f(x)` with
  the exponent raised after it), so that the pretty printer makes the
  context of its base, of the operand of a prefix sign and of the body of
  a sum sign with `Context.at`;
* the pretty printer keeps its rules in tables of `rule_for(cls, rule)`
  entries which `rule_table` puts into the `Dispatcher`, the LaTeX
  printer registers them with the decorator `RULES.of(cls)`.

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

The pretty printer takes a rule in the same way:

```python
>>> from sympy_extras.printing.pretty import PrettyContext, PrettyResult
>>> def _norm_picture(printer: PrettyPrinter, e: Norm, context: PrettyContext) -> PrettyResult:
...     inside = printer.operand(e.args[0], context.delimited())
...     return PrettyResult(printer.parens(inside, '|', '|').ltext('|').rtext('|'), Binding.ATOM)
>>> PRETTY_RULES.register(Norm, _norm_picture)
>>> pprint(Norm(x + y)**2/2)  # doctest: +NORMALIZE_WHITESPACE
         2
||x + y||
----------
    2

```

## Tests

LaTeX:

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

Pretty printing:

* `test_sympy_agree.py`: 359 expressions printed by this printer and
  by SymPy (through `to_sympy(e, evaluate=False)`), in Unicode and in ASCII:
  the pictures are equal character by character. The ASCII output is ASCII.
* `test_semantics.py`: a reader of pictures is hard to write, so all the
  trees of small grammars of unevaluated nodes (8164 trees of depth two over
  `x`, `2`, `-1`, `1/2` with `+`, `*`, `**`; quotients of quotients; sum and
  product signs among factors and terms; formulas with relations and
  connectives; set expressions) and seeded random larger trees with their
  regroupings are printed, and two trees with the same picture must have
  the same value at random points, computed by an evaluator which knows
  nothing of the printer. Disabling any one mechanism of `needs_brackets`
  (strictness, the stacked fractions, the open-ended operands) makes
  one of these tests fail.
* `test_sympy_issue_30844.py`: one test per item above.
* `test_pretty.py`: the settings, the containers, matrices, polynomials,
  pictures, symbol names, and that every class of nodes has a rule.


Both:

* `test_framework.py`: the framework on a toy printer of its own;
* `test_unified.py`: both printers on the one framework, the shared
  conventions, `ITE`, `full_prec`, the interactive session.

## Differences from SymPy

### LaTeX

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

### Pretty printing

Where SymPy's picture reads as another expression, this printer differs
(the tests of `test_sympy_issue_30844.py` quote SymPy's output):

```python
>>> from sympy_extras.core import Mul, Integer, Float
>>> from sympy_extras.core.mod import Mod
>>> from sympy_extras.core.limits import Limit
>>> pprint(Pow(Rational(2, 3), n))  # doctest: +NORMALIZE_WHITESPACE
     n
(2/3)
>>> pprint(Pow(Float(-2.5), n))  # doctest: +NORMALIZE_WHITESPACE
      n
(-2.5)
>>> print(pretty(Mod(x, y + n)), '|', pretty(Mod(-x, y)), '|', pretty(-Mod(x, y)))
x mod (n + y) | (-x) mod y | -(x mod y)
>>> print(pretty(Not(x < y), use_unicode=True), '|', pretty(Mul(x, Integer(-1), Integer(2))))
¬(x < y) | x*(-1)*2
>>> pprint(Limit(sin(x), x, y + n), use_unicode=True)  # doctest: +NORMALIZE_WHITESPACE
    lim    sin(x)
x─→(n + y)⁺
>>> pprint(Mul(Sum(1/k, (k, 1, n)), x), use_unicode=True)  # doctest: +NORMALIZE_WHITESPACE
⎛ n     ⎞
⎜____   ⎟
⎜╲      ⎟
⎜ ╲     ⎟
⎜  ╲   1⎟
⎜  ╱   ─⎟⋅x
⎜ ╱    k⎟
⎜╱      ⎟
⎜‾‾‾‾   ⎟
⎝k = 1  ⎠

```

| Item of sympy/sympy#30844 | SymPy | here |
|---|---|---|
| root cause 1: a negative product | `-x mod y` for both `Mod(-x, y)` and `-Mod(x, y)`; `x⋅-2` | `(-x) mod y`, `-(x mod y)`, `x⋅(-1)⋅2` |
| root cause 3: sides of a relation | `x = y = y = z` | a side binding as a relation or less is bracketed |
| root cause 4: `Not` | `¬t < x`, `¬x ∈ [0, 1]` | `¬(t < x)`, `¬(x ∈ [0, 1])` |
| root cause 5: composite atoms | `2/3` with the exponent `x`, `-1/2ⁿ + x`, `-2.5ⁿ` | `(2/3)ˣ`, `(-1/2)ⁿ + x`, `(-2.5)ⁿ` |
| root cause 6: direction of a limit | `x─→a + b⁺` | `x─→(a + b)⁺` |
| a sum sign captures the next factors | `Product*Sum` puts the sum in the body of the product | a factor open to the right is bracketed unless it is the last |
| `Mod` | `x mod y + z`, `x mod y mod z` | `x mod (y + z)`, `x mod (y mod z)`, `(x mod y) + z` |
| set operations | `X \ (Y \ Z)` printed like `(X \ Y) \ Z`; `X ∪ Y²` | every nested set operation is bracketed |
| `Pow(sin(x)**2, 3)` | `sin²(x)` with a stray `3` | `(sin²(x))³` |
| `Limit(1/x, x, oo)` | `x─→∞x` | the operand is bracketed or kept apart |
| `Product(k + 1, ...)` | printed like `Product(k, ...) + 1` | `∏ (k + 1)` |
| ASCII `SingularityFunction` | garbled | `<-a + x>` raised to `n`, the angle brackets as tall as the shift |
| function names | `reducedₜₒₜᵢₑₙₜ(n)`, `divisor_σ(n)` | the names of defined functions are printed as they are |
| lost information | `Symbol('x_') + x` → `x + x`; `exp_polar(x)` like `exp(x)` | `x + x_`; `exp_polar(x)` |
| crashes | `pretty(Array([]))`, a sequence with symbolic bounds | `[]`; the constructor form |

Other deliberate differences: a nested unevaluated sum keeps its brackets
(`a + (b + c)`); a `Wild` is printed with its underscore (`w_`); `Contains`
in ASCII prints its arguments with the pretty printer; the condition of a
`ConditionSet` is bracketed only when it is a formula with a connective; an
integral of an integral is drawn as a multiple integral, a sum of a sum as a
multiple sum; an infinite end of an interval is drawn open.

## Interactive sessions

```python
>>> from sympy_extras.printing import init_printing
>>> from sympy_extras.printing.interactive import reset_printing, text_form
>>> print((x**2/2)._repr_latex_())
$\displaystyle \frac{x^{2}}{2}$
>>> print(text_form(x**2/2))
x**2/2
>>> init_printing(mul_symbol='dot', display_hook=False)
>>> print((2*x)._repr_latex_())
$\displaystyle 2 \cdot x$
>>> print(text_form(x**2/2))  # doctest: +NORMALIZE_WHITESPACE
 2
x
──
2
>>> init_printing(use_unicode=False, display_hook=False)
>>> print(text_form(x**2/2))  # doctest: +NORMALIZE_WHITESPACE
 2
x
--
2
>>> init_printing(pretty_print=False, use_latex=False, display_hook=False)
>>> print(text_form(x**2/2), (2*x)._repr_latex_())
x**2/2 None
>>> reset_printing()

```

`init_printing(pretty_print=True, use_unicode=True, use_latex=True, ...)`
chooses how the native objects are shown:

* `use_latex`: `_repr_latex_` returns the LaTeX (with the settings of
  `latex` given to `init_printing`), which a notebook renders; `False`
  makes it return `None`, so that the notebook shows the text;
* `pretty_print`: the text of a result (`text_form`: what `_repr_pretty_`
  writes for IPython and what the display hook of a plain Python session
  prints) is the two-dimensional form of the pretty printer, with the
  order of the terms, the notation of the roots, the imaginary unit and
  `full_prec` chosen for the LaTeX where the pretty printer has them;
  `False` shows the string form. Before `init_printing` is called the
  text is the string form;
* `use_unicode`: pictures in Unicode or in ASCII; it becomes the default
  of `pretty` and `pprint` too (`pretty_use_unicode`);
* `display_hook`: in a plain Python session `init_printing()` installs a
  display hook which prints the native objects in their text form.

`sympy_extras.printing.interactive.set_pretty_printer` plugs in another
two-dimensional printer (a function of the object and of whether Unicode
may be used) in the place of the one of this package; `reset_printing()`
undoes `init_printing`. `ExprMatrix`, `Poly` and `DomainMatrix` (shown as
the matrix of its elements as expressions) have the two methods too.
Nothing of IPython is imported or configured.

## References

* SymPy Development Team, *SymPy documentation: Printing*,
  <https://docs.sympy.org/latest/modules/printing.html>.
* sympy/sympy#30844, *Printers: systematic review of precedence and other
  defects*, <https://github.com/sympy/sympy/issues/30844>.
