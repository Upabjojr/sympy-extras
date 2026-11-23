# Pretty printing

`sympy_extras.printing.pretty` draws the expressions of the typed core as
two-dimensional text, in Unicode or in ASCII: fraction bars, raised
exponents, root signs, sum, product and integral signs, braces of cases. It
is a typed port of SymPy's pretty printer (`sympy/printing/pretty`), which
it follows character by character where SymPy is right, with another design
of the dispatch and of the bracketing. The defects of the pretty printer
listed in [sympy/sympy#30844](https://github.com/sympy/sympy/issues/30844)
are not reproduced.

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

## Functions

| Function | |
|---|---|
| `pretty(expr, use_unicode=None, wrap_line=True, num_columns=None, order=None, root_notation=True, use_unicode_sqrt_char=True, imaginary_unit='i')` | the picture as a string; its rows have the same width |
| `pprint(expr, ...)`, `pretty_print` | prints it |
| `pretty_picture(expr, settings)` | the `Picture` (rows and baseline), not cut into screens |
| `pretty_use_unicode(flag=None)` | the default of `use_unicode` (ASCII at the start); returns the previous value |
| `pretty_try_use_unicode()` | makes Unicode the default when the standard output can encode it |
| `pretty_repr(expr, sink, cycle, settings=None)` | the body of an IPython `_repr_pretty_(self, p, cycle)` method, for `init_printing` |

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

```

## Design

SymPy's printer finds a method `_print_<ClassName>` by the *name* of the
class, and a parent decides on the brackets of a child from the class of the
child (a precedence number), not from what the child printed. Most of the
pretty section of sympy/sympy#30844 comes from that: a `prettyForm` carries
a binding, but many rules build a composite picture without setting it, so
that it stays `ATOM` (root cause 5). Here:

* **Dispatch by class.** `RULES` is a table of `RuleEntry(cls, rule)`; the
  printer takes the entry of the most specific class of the node
  (`isinstance`, no lookup by name, no reflection). A rule is a function
  `(printer, node, context) -> PrettyResult`. A virtual method on the nodes,
  as `_sympystr` is, was the other option: the table was chosen because the
  printer then stays out of the core (the nodes do not know of pictures, and
  a third printer needs no change of them), and because it is a `match` on
  the kind of the node in a statically typed language.
* **Typed results.** A rule returns a `PrettyResult`: the `Picture`, the
  `Binding` of its outermost operator *as printed* (no default: a rule must
  say it), and the flags a parent needs: `magnitude` (the picture without
  its leading minus sign, so that a sum writes `a - b`), `open_right` (the
  picture ends with the operand of a sum sign, a product sign or a limit),
  `stacked` (the outermost layout is a fraction bar), `applied` (the head
  and the arguments of a function application, between which a power puts
  its exponent).
* **Typed contexts.** A parent prints a child with
  `printer.operand(child, context.child(position, minimum, ...))`. The
  `Context` is immutable: the `Position` of the child (operand of an infix
  operator and which one, base, exponent, numerator, side of a relation,
  body of a sum sign, point of a limit, ...), the binding the position
  requires, and the inherited state (depth, inside an exponent, fractions
  stacked). The settings are never changed during the descent.
* **Brackets in one place.** `needs_brackets(result, context)` decides, and
  `PrettyPrinter.bracketed` is the only code which adds parentheses. (The
  delimiters of a notation, `f(...)`, `|x|`, `d/dx(...)`, are not brackets
  of precedence.)

The bindings, from the loosest to the tightest:

| `Binding` | printed forms |
|---|---|
| `OPEN` | `x ↦ y`, a brace of cases |
| `IMPLIES`, `XOR`, `OR`, `AND` | the connectives |
| `RELATION` | `=`, `<`, `∈` |
| `SET` | `∪`, `∩`, `\`, `×` |
| `MOD` | `x mod y` |
| `ADD` | sums |
| `NEG` | what starts with a minus sign: `-x`, `-2`, `-1/2`, `-2.5` |
| `INLINE_QUOTIENT` | a fraction on one row: `1/2` |
| `MUL` | products, sum and product signs, integrals, limits, derivatives |
| `DIV` | a fraction with a bar |
| `ROOT` | `√x` |
| `POW` | powers, `sin²(x)` |
| `AFFIX` | `¬x`, `x!`, `∀x ...` |
| `FUNC` | `f(x)` |
| `ATOM` | one token, or a form closed by its own delimiters |

```python
>>> from sympy_extras.printing.pretty import PrettyPrinter, PrettySettings, RULES, TOP
>>> printer = PrettyPrinter(PrettySettings(), RULES)
>>> [printer.result(e, TOP).binding.name for e in (x, -x, x/y, Rational(1, 2), sin(x), sin(x)**2)]
['ATOM', 'NEG', 'DIV', 'INLINE_QUOTIENT', 'FUNC', 'POW']

```

## Differences from SymPy

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

## Tests

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
  (strictness, the stacked fractions, the operands open to the right) makes
  one of these tests fail.
* `test_sympy_issue_30844.py`: one test per item above.
* `test_pretty.py`: the settings, the containers, matrices, polynomials,
  pictures, symbol names, and that every class of nodes has a rule.

## References

* SymPy Development Team, *SymPy documentation: Printing*,
  <https://docs.sympy.org/latest/modules/printing.html>.
* sympy/sympy#30844, *Printers: systematic review of precedence and other
  defects*, <https://github.com/sympy/sympy/issues/30844>.
