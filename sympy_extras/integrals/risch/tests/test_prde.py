"""Most of these tests come from the examples in Bronstein's book (on the
typed core: the polynomials of integrals.risch.tower, the matrices of
integrals.risch.polymatrix)."""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.testing.pytest import raises

from sympy_extras.core import I, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import exp
from sympy_extras.integrals.risch.risch import DifferentialExtension, Extension, derivation, frac_in
from sympy_extras.integrals.risch.rde import OO
from sympy_extras.integrals.risch.prde import (prde_normal_denom, prde_special_denom,
    prde_linear_constraints, constant_system, prde_spde, prde_no_cancel_b_large,
    prde_no_cancel_b_small, prde_no_cancel_b_equal, limited_integrate_reduce,
    limited_integrate,
    is_deriv_k, is_log_deriv_k_t_radical, parametric_log_deriv_heu,
    parametric_log_deriv, parametric_log_deriv_structure,
    is_log_deriv_k_t_radical_in_field, param_poly_rischDE, param_rischDE,
    is_deriv_in_field, is_deriv_k_atan, is_log_deriv_k_t_radical_tan,
    prde_cancel_liouvillian, prde_cancel_tan)
from sympy_extras.integrals.risch.polymatrix import FieldMatrix
from sympy_extras.integrals.risch.tower import RationalFunction, TowerPoly

x, t, n, y = Symbol('x'), Symbol('t'), Symbol('n'), Symbol('y')
t0, t1, t2 = Symbol('t0'), Symbol('t1'), Symbol('t2')

Like = object


def R(e: Like) -> RationalFunction:
    if isinstance(e, RationalFunction):
        return e
    if isinstance(e, int):
        return RationalFunction.from_int(e)
    assert isinstance(e, Expr), e
    return RationalFunction.from_expr(e)


def P(e: Like, gen: Expr) -> TowerPoly:
    return TowerPoly.from_element_strict(R(e), gen)


def M(rows: Sequence[Sequence[Like]]) -> FieldMatrix:
    return FieldMatrix.from_rows([[R(e) for e in r] for r in rows])


def col(entries: Sequence[Like]) -> FieldMatrix:
    return FieldMatrix.column([R(e) for e in entries])


def DE_of(D: Sequence[TowerPoly], exts: Sequence[str] = (), extargs: Sequence[Expr] = (),
          labeled: bool = False) -> DifferentialExtension:
    if not labeled:
        return DifferentialExtension.manual(list(D))
    ex: list[Optional[Extension]] = [_ext(e) for e in exts]
    return DifferentialExtension.manual(list(D), exts=ex, extargs=list(extargs))


def _ext(e: str) -> Extension:
    if e == 'exp':
        return 'exp'
    if e == 'log':
        return 'log'
    if e == 'tan':
        return 'tan'
    assert e == 'atan'
    return 'atan'


def fr(f: tuple[TowerPoly, TowerPoly]) -> RationalFunction:
    return f[0].as_element()/f[1].as_element()


def test_prde_normal_denom() -> None:
    DE = DE_of([P(1, x), P(1 + t**2, t)])
    fa = P(1, t)
    fd = P(x, t)
    G = [(P(t, t), P(1 + t**2, t)), (P(1, t), P(x + x*t**2, t))]
    assert prde_normal_denom(fa, fd, G, DE) == \
        (P(x, t), (P(1, t), P(1, t)), [(P(x*t, t), P(t**2 + 1, t)), (P(1, t), P(t**2 + 1, t))], P(1, t))
    G = [(P(t, t), P(t**2 + 2*t + 1, t)), (P(x*t, t),
        P(t**2 + 2*t + 1, t)), (P(x*t**2, t), P(t**2 + 2*t + 1, t))]
    DE = DE_of([P(1, x), P(t, t)])
    assert prde_normal_denom(P(x, t), P(1, t), G, DE) == \
        (P(t + 1, t), (P((-1 + x)*t + x, t), P(1, t)), [(P(t, t),
        P(1, t)), (P(x*t, t), P(1, t)), (P(x*t**2, t),
        P(1, t))], P(t + 1, t))


def test_prde_special_denom() -> None:
    a = P(t + 1, t)
    ba = P(t**2, t)
    bd = P(1, t)
    G = [(P(t, t), P(1, t)), (P(t**2, t), P(1, t)), (P(t**3, t), P(1, t))]
    DE = DE_of([P(1, x), P(t, t)])
    assert prde_special_denom(a, ba, bd, G, DE) == \
        (P(t + 1, t), P(t**2, t), [(P(t, t), P(1, t)),
        (P(t**2, t), P(1, t)), (P(t**3, t), P(1, t))], P(1, t))
    G = [(P(t, t), P(1, t)), (P(1, t), P(t, t))]
    assert prde_special_denom(P(1, t), P(t**2, t), P(1, t), G, DE) == \
        (P(1, t), P(t**2 - 1, t), [(P(t**2, t), P(1, t)),
        (P(1, t), P(1, t))], P(t, t))
    DE = DE_of([P(1, x), P(-2*x*t0, t0)])
    DE.decrement_level()
    G = [(P(t, t), P(t**2, t)), (P(2*t, t), P(t, t))]
    assert prde_special_denom(P(5*x*t + 1, t), P(t**2 + 2*x**3*t, t), P(t**3 + 2, t), G, DE) == \
        (P(5*x*t + 1, t), P(0, t), [(P(t, t), P(t**2, t)),
        (P(2*t, t), P(t, t))], P(1, x))
    DE = DE_of([P(1, x), P((t**2 + 1)*2*x, t)])
    G = [(P(t + x, t), P(t*x, t)), (P(2*t, t), P(x**2, x))]
    # the old Poly(2*t, t, x) entries: polynomials in t
    assert prde_special_denom(P(5*x*t + 1, t), P(t**2 + 2*x**3*t, t), P(t**3, t), G, DE) == \
        (P(5*x*t + 1, t), P(0, t), [(P(t + x, t), P(x*t, t)),
        (P(2*t, t), P(x**2, t))], P(1, t))
    assert prde_special_denom(P(t + 1, t), P(t**2, t), P(t**3, t), G, DE) == \
        (P(t + 1, t), P(0, t), [(P(t + x, t), P(x*t, t)), (P(2*t, t),
        P(x**2, t))], P(1, t))

    # Hypertangent case with a nontrivial special denominator (parametric
    # version of the corresponding test in test_rde.py)
    DE = DE_of([P(1, x), P(t**2 + 1, t)])
    assert prde_special_denom(P(t, t), P(t, t), P(1, t),
        [(P(-2*t**2 + t, t), P(t**2 + 1, t))], DE) == \
        (P(t, t), P(-2*t**2 + t, t),
        [(P(-2*t**2 + t, t), P(1, t))], P(t**2 + 1, t))


def test_prde_linear_constraints() -> None:
    DE = DE_of([P(1, x)])
    G = [(P(2*x**3 + 3*x + 1, x), P(x**2 - 1, x)), (P(1, x), P(x - 1, x)),
        (P(1, x), P(x + 1, x))]
    assert prde_linear_constraints(P(1, x), P(0, x), G, DE) == \
        ((P(2*x, x), P(0, x), P(0, x)), M([[1, 1, -1], [5, 1, 1]]))
    G = [(P(t, t), P(1, t)), (P(t**2, t), P(1, t)), (P(t**3, t), P(1, t))]
    DE = DE_of([P(1, x), P(t, t)])
    assert prde_linear_constraints(P(t + 1, t), P(t**2, t), G, DE) == \
        ((P(t, t), P(t**2, t), P(t**3, t)), FieldMatrix(0, 3, []))
    G = [(P(2*x, t), P(t, t)), (P(-x, t), P(t, t))]
    DE = DE_of([P(1, x), P(1/x, t)])
    assert prde_linear_constraints(P(1, t), P(0, t), G, DE) == \
        ((P(0, t), P(0, t)), M([[2*x, -x]]))


def test_constant_system() -> None:
    A = M([[-(x + 3)/(x - 1), (x + 1)/(x - 1), 1],
           [-x - 3, x + 1, x - 1],
           [2*(x + 3)/(x - 1), 0, 0]])
    u = col([(x + 1)/(x - 1), x + 1, 0])
    DE = DE_of([P(1, x)])
    assert constant_system(A, u, DE) == \
        (M([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1]]), col([0, 1, 0, 0]))

    # Multiple rows with symbolic constants: pivoting on y is sound, since
    # y is a nonzero element of the constant field QQ(y)
    A = M([[y, 1, x], [0, y, 1 + x]])
    u = col([x + y, x + 2])
    B, v = constant_system(A, u, DE)
    assert (B, v) == (M([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), col([(y**2 - 1)/y**2, 1/y, 1]))

    # An entry whose derivation-quotient rows still contain tower
    # variables, requiring more than one pass of the elimination loop
    # ("while A is not constant" in the book's ConstantSystem); the
    # output must be fully constant
    DE = DE_of([P(1, x), P(t0, t0), P((t0 + x*t0)*t1, t1)], ['exp', 'exp'], [x, x*t0], labeled=True)
    A = M([[t1, t0]])
    u = col([0])
    B, v = constant_system(A, u, DE)
    assert not any(B[i, j].has(t0, t1) for i in range(B.rows) for j in range(B.cols))


def test_prde_spde() -> None:
    D = [P(x, t), P(-x*t, t)]
    DE = DE_of([P(1, x), P(1/x, t)])
    # TODO: when bound_degree() can handle this, test degree bound from that too
    # (the old test passed the symbol n as the degree bound; any integer
    # shows the same shift by deg(a))
    assert prde_spde(P(t, t), P(-1/x, t), D, 5, DE) == \
        (P(t, t), P(0, t), [P(2*x, t), P(-x, t)], [P(-x**2, t), P(0, t)], 4)


def test_prde_no_cancel() -> None:
    # b large
    DE = DE_of([P(1, x)])
    assert prde_no_cancel_b_large(P(1, x), [P(x**2, x), P(1, x)], 2, DE) == \
        ([P(x**2 - 2*x + 2, x), P(1, x)], M([[1, 0, -1, 0],
                                             [0, 1, 0, -1]]))
    assert prde_no_cancel_b_large(P(1, x), [P(x**3, x), P(1, x)], 3, DE) == \
        ([P(x**3 - 3*x**2 + 6*x - 6, x), P(1, x)], M([[1, 0, -1, 0],
                                                      [0, 1, 0, -1]]))
    assert prde_no_cancel_b_large(P(x, x), [P(x**2, x), P(1, x)], 1, DE) == \
        ([P(x, x), P(0, x)], M([[1, -1,  0,  0],
                                [1,  0, -1,  0],
                                [0,  1,  0, -1]]))
    # b small
    # XXX: Is there a better example of a monomial with D.degree() > 2?
    DE = DE_of([P(1, x), P(t**3 + 1, t)])

    # My original q was t**4 + t + 1, but this solution implies q == t**4
    # (c1 = 4), with some of the ci for the original q equal to 0.
    G = [P(t**6, t), P(x*t**5, t), P(t**3, t), P(x*t**2, t), P(1 + x, t)]
    assert prde_no_cancel_b_small(P(x*t, t), G, 4, DE) == \
        ([P(t**4/4 - x/12*t**3 + x**2/24*t**2 + (Rational(-11, 12) - x**3/24)*t + x/24, t),
        P(x/3*t**3 - x**2/6*t**2 + (Rational(-1, 3) + x**3/6)*t - x/6, t), P(t, t),
        P(0, t), P(0, t)], M([[1, 0,              -1, 0, 0,  0,  0,  0,  0,  0],
                              [0, 1, Rational(-1, 4), 0, 0,  0,  0,  0,  0,  0],
                              [0, 0,               0, 0, 0,  0,  0,  0,  0,  0],
                              [0, 0,               0, 1, 0,  0,  0,  0,  0,  0],
                              [0, 0,               0, 0, 1,  0,  0,  0,  0,  0],
                              [1, 0,               0, 0, 0, -1,  0,  0,  0,  0],
                              [0, 1,               0, 0, 0,  0, -1,  0,  0,  0],
                              [0, 0,               1, 0, 0,  0,  0, -1,  0,  0],
                              [0, 0,               0, 1, 0,  0,  0,  0, -1,  0],
                              [0, 0,               0, 0, 1,  0,  0,  0,  0, -1]]))

    # TODO: Add test for deg(b) <= 0 with b small
    DE = DE_of([P(1, x), P(1 + t**2, t)])
    b = P(-1/x**2, t)  # deg(b) == 0
    q = [P(x**i*t**j, t) for i in range(2) for j in range(3)]
    h, A = prde_no_cancel_b_small(b, list(q), 3, DE)
    V = A.nullspace()
    assert len(V) == 1
    assert V[0] == col([Rational(-1, 2), 0, 0, 1, 0, 0]*3)
    assert (FieldMatrix.row_vector([hi.as_element() for hi in h])*V[0][6:, :]).element(0) == R(x**2/2)
    assert (FieldMatrix.row_vector([qi.as_element() for qi in q])*V[0][:6, :]).element(0) == R(x - Rational(1, 2))


def _combination(v: FieldMatrix, start: int, h: Sequence[TowerPoly]) -> RationalFunction:
    total = RationalFunction.from_int(0)
    for j in range(len(h)):
        total = total + v[start + j, 0]*h[j].as_element()
    return total


def test_prde_no_cancel_b_equal() -> None:
    # deg(b) == delta(t) - 1, with t == tan-like (Dt == t**2 + 1)
    DE = DE_of([P(1, x), P(t**2 + 1, t)])
    # -lc(b)/lc(Dt) == -1 is not a positive integer, so the loop runs to
    # N == 0: Dq + t*q == 2*t**2 + 1 has the solution q == t
    assert prde_no_cancel_b_equal(P(t, t), [P(2*t**2 + 1, t)], 1, DE) == \
        ([P(t, t)], M([[1, -1]]))
    # A solution with a nonconstant coefficient and two right hand sides:
    # q == t**2 solves Dq + t*q == c1 for c == D(t**2) + t*t**2
    b = P(t, t)
    q = P(t**2, t)
    c = derivation(q, DE) + b*q
    h, A = prde_no_cancel_b_equal(b, [c, P(0, t)], 2, DE)
    V = A.nullspace()
    sols = 0
    for v in V:
        if v[0, 0]:
            yp = P(_combination(v, 2, h)/v[0, 0], t)
            if (derivation(yp, DE) + b*yp - c).is_zero:
                sols += 1
    assert sols >= 1
    # When the possible-cancellation degree -lc(b)/lc(Dt) is reached, the
    # rest is delegated to the cancellation algorithms (this used to
    # raise NotImplementedError before prde_cancel_tan() existed):
    # Dq - 2*t*q == c1*(t**4 + 3*t**2) has the solutions
    # q == c1*t**3 + d*(t**2 + 1) (t**2 + 1 solves the homogeneous
    # equation)
    b = P(-2*t, t)
    Q = [P(t**4 + 3*t**2, t)]
    h, A = prde_no_cancel_b_equal(b, Q, 3, DE)
    matched: set[tuple[RationalFunction, TowerPoly]] = set()
    for v in A.nullspace():
        c1 = v[0, 0]
        ysol = P(_combination(v, 1, h), t)
        assert (derivation(ysol, DE) + b*ysol - Q[0].mul_ground(c1)).is_zero
        matched.add((c1, ysol))
    assert (R(1), P(t**3, t)) in matched
    assert (R(0), P(t**2 + 1, t)) in matched
    # A cancellation degree above the bound n is unattainable, so the
    # descent runs through N == 0 instead of handing off with a bound
    # above the caller's: Dq + (1 - 2*t)*q == c1*(-t**2 + t + 1) with
    # n == 1 has the solutions q == c1*t
    b = P(1 - 2*t, t)
    Q = [P(-t**2 + t + 1, t)]
    h, A = prde_no_cancel_b_equal(b, Q, 1, DE)
    matched = set()
    for v in A.nullspace():
        c1 = v[0, 0]
        ysol = P(_combination(v, 1, h), t)
        assert (derivation(ysol, DE) + b*ysol - Q[0].mul_ground(c1)).is_zero
        matched.add((c1, ysol))
    assert (R(1), P(t, t)) in matched


def test_prde_cancel_liouvillian() -> None:
    ### 1. case == 'primitive'
    # used when integrating f = log(x) - log(x - 1)
    # Not taken from 'the' book
    DE = DE_of([P(1, x), P(1/x, t)])
    p0 = P(0, t)
    p1 = P((x - 1)*t, t)
    p2 = P(x - 1, t)
    p3 = P(-x**2 + x, t)
    h, A = prde_cancel_liouvillian(P(-1/(x - 1), t), [P(-x + 1, t), P(1, t)], 1, DE)
    V = A.nullspace()
    assert h == [p0, p0, p1, p0, p0, p0, p0, p0, p0, p0, p2, p3, p0, p0, p0, p0]
    assert A.rank() == 16
    assert (FieldMatrix.row_vector([hi.as_element() for hi in h])*V[0][:16, :]) == M([[0]])

    ### 2. case == 'exp'
    # used when integrating log(x/exp(x) + 1)
    # Not taken from book
    DE = DE_of([P(1, x), P(-t, t)])
    assert prde_cancel_liouvillian(P(0, t), [P(1, t)], 0, DE) == \
            ([P(1, t), P(x, t)], M([[-1, 0, 1]]))

    ### 3. case == 'exp' with b != 0 and n > 0.  This exercises the level at
    ### which eta == Dt/t is computed and the sign of the residual update
    ### Fi == -(D(h) + b*h), both of which used to be wrong.
    DE = DE_of([P(1, x), P(t, t)])
    b = P(1/x, t)
    Q = [P((x + 1)*t/x, t)]
    h, A = prde_cancel_liouvillian(b, list(Q), 1, DE)
    V = A.nullspace()
    # Dy + y/x == c1*(x + 1)*t/x must have the solution y == t (c1 == 1)
    found = False
    for v in V:
        if not v[0, 0]:
            continue
        yp = P(_combination(v, 1, h)/v[0, 0], t)
        if (derivation(yp, DE) + b*yp - Q[0]).is_zero:
            found = True
    assert found


def test_prde_cancel_tan() -> None:
    # t = tan(x)
    DE = DE_of([P(1, x), P(t**2 + 1, t)])
    b = P(1 - t, t)
    Q = [P(t**3 + t**2 - 2*x*t - 2*x, t)]

    def solutions(h: list[TowerPoly], A: FieldMatrix, b: TowerPoly, Q: list[TowerPoly]
            ) -> list[tuple[list[RationalFunction], TowerPoly]]:
        # Every nullspace vector of A must produce a solution of
        # Dq + b*q == Sum(ci*qi); return the (c, q) pairs.
        sols: list[tuple[list[RationalFunction], TowerPoly]] = []
        for v in A.nullspace():
            cs = [v[i, 0] for i in range(len(Q))]
            q = P(_combination(v, len(Q), h), t)
            lhs = derivation(q, DE) + b*q
            rhs = P(0, t)
            for ci, qi in zip(cs, Q):
                rhs = rhs + qi.mul_ground(ci)
            assert (lhs - rhs).is_zero
            sols.append((cs, q))
        return sols

    # The parametric version of Example 6.6.1, reached through the
    # full param_poly_rischDE() hand-off (the parametric Example
    # 6.5.3): Dq + (1 - t)*q == c1*(t**3 + t**2 - 2*x*t - 2*x) with
    # n == 2 has exactly the solutions q == c1*(t**2 - 2*x*t)
    h, A = param_poly_rischDE(P(1, t), b, list(Q), 2, DE)
    sols = solutions(h, A, b, Q)
    assert any(cs == [1] and q == P(t**2 - 2*x*t, t) for cs, q in sols)

    # Homogeneous solutions are found: D(t**2 + 1) == 2*t*(t**2 + 1),
    # so q == t**2 + 1 solves Dq - 2*t*q == 0 (b0 == 0, n == 2)
    b0 = P(0, t)
    h, A = prde_cancel_tan(b0, [P(0, t)], 2, DE)
    sols = solutions(h, A, P(-2*t, t), [P(0, t)])
    assert any(q == P(t**2 + 1, t) for cs, q in sols)

    # n == 0: q in k, so the t-coefficients of the right hand side
    # must vanish: Dq + q == c1*(x + 1) + c2*t forces c2 == 0, with
    # the solution q == c1*x
    h, A = prde_cancel_tan(P(1, t), [P(x + 1, t), P(t, t)], 0, DE)
    sols = solutions(h, A, P(1, t), [P(x + 1, t), P(t, t)])
    assert sols and all(cs[1] == 0 for cs, q in sols)
    assert any(cs == [1, 0] and q == P(x, t) for cs, q in sols)

    # The degree bound appears in the equation itself, so n == oo is
    # not allowed
    raises(ValueError, lambda: prde_cancel_tan(P(1, t), [P(1, t)],
        OO, DE))


def test_param_poly_rischDE() -> None:
    DE = DE_of([P(1, x)])
    a = P(x**2 - x, x)
    b = P(1, x)
    q = [P(x, x), P(x**2, x)]
    h, A = param_poly_rischDE(a, b, q, 3, DE)

    assert A.nullspace() == [col([0, 1, 1, 1])]  # c1, c2, d1, d2
    # Solution of a*Dp + b*p = c1*q1 + c2*q2 = q2 = x**2
    # is d1*h1 + d2*h2 = h1 + h2 = x.
    assert h[0] + h[1] == P(x, x)
    # a*Dp + b*p = q1 = x has no solution.

    a = P(x**2 - x, x)
    b = P(x**2 - 5*x + 3, x)
    q = [P(1, x), P(x, x), P(x**2, x)]
    h, A = param_poly_rischDE(a, b, q, 3, DE)

    assert A.nullspace() == [col([3, -5, 1, -5, 1, 1])]
    p = -h[0].mul_ground(5) + h[1] + h[2]  # Poly(1, x)
    assert a*derivation(p, DE) + b*p == P(x**2 - 5*x + 3, x)


def test_param_rischDE() -> None:
    DE = DE_of([P(1, x)])
    p1, px = P(1, x), P(x, x)
    G = [(p1, px), (p1, p1), (px, p1)]  # [1/x, 1, x]
    h, A = param_rischDE(-p1, P(x**2, x), G, DE)
    assert len(h) == 3
    p = [fr(hi) for hi in h]
    V = A.nullspace()
    assert len(V) == 2
    assert V[0] == col([-1, 1, 0, -1, 1, 0])
    yv = -p[0] + p[1] + 0*p[2]  # x
    assert yv.diff(x) - yv/R(x**2) == R(1 - 1/x)  # Dy + f*y == -G0 + G1 + 0*G2

    # the below test computation takes place while computing the integral
    # of 'f = log(log(x + exp(x)))'
    DE = DE_of([P(1, x), P(t, t)])
    G = [(P(t + x, t), P(1, t)), (P(0, t), P(1, t))]
    h, A = param_rischDE(P(-t - 1, t), P(t + x, t), G, DE)
    assert len(h) == 5
    p = [fr(hi) for hi in h]
    V = A.nullspace()
    assert len(V) == 3
    assert V[0] == col([0, 0, 0, 0, 1, 0, 0])
    yv = 0*p[0] + 0*p[1] + 1*p[2] + 0*p[3] + 0*p[4]
    assert yv.diff(t) - yv/R(t + x) == 0   # Dy + f*y = 0*G0 + 0*G1


def test_limited_integrate_reduce() -> None:
    DE = DE_of([P(1, x), P(1/x, t)])
    assert limited_integrate_reduce(P(x, t), P(t**2, t), [(P(x, t),
    P(t, t))], DE) == \
        (P(t, t), P(-1/x, t), P(t, t), 1, (P(x, t), P(1, t)),
        [(P(-x*t, t), P(1, t))])

    # An exp case with a nontrivial special part (hs != 1).  This
    # distinguishes the first return component, which must be hn rather
    # than a == hn*hs (the book returns a, which does not satisfy the
    # stated contract): check the contract directly for the known
    # solution v == 1/(t*x), c1 == 3 of f == Dv + c1*w1.
    DE = DE_of([P(1, x), P(t, t)])
    fa, fd = frac_in((3*x**2*t - x - 1)/(x**2*t), t)
    G = [(P(1, t), P(1, t))]
    A, b, h, N, g, V = limited_integrate_reduce(fa, fd, G, DE)
    p = P(h.as_element()/R(t*x), t)  # p == v*h
    assert p.degree(t) <= N
    lhs = (A*derivation(p, DE) + b*p).as_element()
    rhs = fr(g) + fr(V[0])*3
    assert lhs == rhs


def test_limited_integrate() -> None:
    DE = DE_of([P(1, x)])
    G = [(P(x, x), P(x + 1, x))]
    assert limited_integrate(P(-(1 + x + 5*x**2 - 3*x**3), x),
    P(1 - x - x**2 + x**3, x), G, DE) == \
        ((P(x**2 - x + 2, x), P(x - 1, x)), [R(2)])
    G = [(P(1, x), P(x, x))]
    assert limited_integrate(P(5*x**2, x), P(3, x), G, DE) == \
        ((P(5*x**3/9, x), P(1, x)), [R(0)])
    # An empty list of special elements (the is_deriv_in_field() case)
    assert limited_integrate(P(2*x, x), P(1, x), [], DE) == \
        ((P(x**2, x), P(1, x)), [])
    # ... and with a symbolic constant coefficient
    assert limited_integrate(P(2*y*x, x), P(1, x), [], DE) == \
        ((P(y*x**2, x), P(1, x)), [])


def test_is_log_deriv_k_t_radical() -> None:
    DE = DE_of([P(1, x)], [], [], labeled=True)
    assert is_log_deriv_k_t_radical(P(2*x, x), P(1, x), DE) is None

    DE = DE_of([P(1, x), P(2*t1, t1), P(1/x, t2)], ['exp', 'log'], [2*x, x], labeled=True)
    assert is_log_deriv_k_t_radical(P(x + t2/2, t2), P(1, t2), DE) == \
        ([(t1, 1), (x, 1)], t1*x, 2, 0)
    # TODO: Add more tests

    DE = DE_of([P(1, x), P(t0, t0), P(1/x, t)], ['exp', 'log'], [x, x], labeled=True)
    assert is_log_deriv_k_t_radical(P(x + t/2 + 3, t), P(1, t), DE) == \
        ([(t0, 2), (x, 1)], x*t0**2, 2, 3)

    # A tower with a symbolic constant parameter: t0 == exp(y*x).
    # exp(2*y*x) == t0**2 is a K-radical; the structure coefficients are
    # rational even though the constant field is QQ(y)
    DE = DE_of([P(1, x), P(y*t0, t0)], ['exp'], [y*x], labeled=True)
    assert is_log_deriv_k_t_radical(P(2*y*x, t0), P(1, t0), DE) == \
        ([(t0, 2)], t0**2, 1, 0)

    # A constant is the logarithmic derivative of the radical 1
    DE = DE_of([P(1, x)], [], [], labeled=True)
    assert is_log_deriv_k_t_radical(P(2, x), P(1, x), DE) == \
        ([], 1, 1, 2)

    # Real towers (Corollary 9.3.2 (ii)): hypertangent and arc-tangent
    # monomials never contribute.  t0 == exp(x), t1 == tan(x),
    # t2 == atan(x); exp(2*x) == t0**2
    DE = DE_of([P(1, x), P(t0, t0), P(1 + t1**2, t1), P(1/(x**2 + 1), t2)],
        ['exp', 'tan', 'atan'], [x, x, x], labeled=True)
    assert is_log_deriv_k_t_radical(P(2*x, t2), P(1, t2), DE) == \
        ([(t0, 2)], t0**2, 1, 0)
    # ... but exp(atan(x)) is a new monomial over it
    assert is_log_deriv_k_t_radical(P(t2, t2), P(1, t2), DE) is None


def test_structure_theorem_guards() -> None:
    # The structure theorems need every monomial of the tower labeled
    # (len(exts) == len(D) - 1)
    for D in ([P(1, x), P(1/x, t1), P(t2, t2)],
              [P(1, x), P(1/x, t1), P(1/(x**2 + 1), t2)]):
        DE = DE_of(D, ['log'], [x], labeled=True)
        for func in (is_deriv_k, is_log_deriv_k_t_radical, is_deriv_k_atan,
                is_log_deriv_k_t_radical_tan):
            try:
                func(P(t2, t2), P(1, t2), DE)
            except NotImplementedError as e:
                assert "labeled" in str(e)
            else:
                raise AssertionError("NotImplementedError was not raised")

    # The real version of the structure theorems, needed as soon as the
    # tower has a hypertangent or arc-tangent monomial, requires sqrt(-1)
    # not in the field: neither in the input ...
    DE = DE_of([P(1, x), P(1/x, t1), P(1 + t2**2, t2)], ['log', 'tan'], [x, x], labeled=True)
    for func in (is_deriv_k, is_log_deriv_k_t_radical, is_deriv_k_atan,
            is_log_deriv_k_t_radical_tan):
        try:
            func(P(I*x, t2), P(1, t2), DE)
        except NotImplementedError as e:
            assert "sqrt(-1)" in str(e)
        else:
            raise AssertionError("NotImplementedError was not raised")
    # ... nor in the tower (t2 == tan(I*x))
    DE = DE_of([P(1, x), P(1/x, t1), P(I*(1 + t2**2), t2)], ['log', 'tan'], [x, I*x], labeled=True)
    raises(NotImplementedError, lambda: is_deriv_k(P(x, t2), P(1, t2), DE))
    # whereas with only exponentials and logarithms, I is a fine constant
    DE = DE_of([P(1, x), P(I*t1, t1)], ['exp'], [I*x], labeled=True)
    assert is_deriv_k(P(t1, t1), P(1, t1), DE) == ([(I*x, 1)], I*x, 1)
    # ... except for the tan/atan adjunction tests, which always use the
    # real theorem: over QQ(I)(x, exp(I*x)), tan(x) is in the field, and
    # over QQ(I)(x, log((1 + I*x)/(1 - I*x))), atan(x) == -I*t/2 is, so
    # a "no solution" answer from the real theorem's empty sums would be
    # wrong
    raises(NotImplementedError, lambda: is_log_deriv_k_t_radical_tan(
        P(x, t1), P(1, t1), DE))
    DE = DE_of([P(1, x), P(2*I/(1 + x**2), t1)], ['log'], [(1 + I*x)/(1 - I*x)], labeled=True)
    raises(NotImplementedError, lambda: is_deriv_k_atan(P(x, t1),
        P(1, t1), DE))
    # and even over QQ(x) when f itself involves I
    DE = DE_of([P(1, x)], [], [], labeled=True)
    raises(NotImplementedError, lambda: is_deriv_k_atan(P(I*x, x),
        P(1, x), DE))


def test_is_deriv_k() -> None:
    DE = DE_of([P(1, x), P(1/x, t1), P(1/(x + 1), t2)], ['log', 'log'], [x, x + 1], labeled=True)
    assert is_deriv_k(P(2*x**2 + 2*x, t2), P(1, t2), DE) == \
        ([(t1, 1), (t2, 1)], t1 + t2, 2)

    DE = DE_of([P(1, x), P(1/x, t1), P(t2, t2)], ['log', 'exp'], [x, x], labeled=True)
    assert is_deriv_k(P(x**2*t2**3, t2), P(1, t2), DE) == \
        ([(x, 3), (t1, 2)], 2*t1 + 3*x, 1)
    # TODO: Add more tests, including ones with exponentials

    DE = DE_of([P(1, x), P(2/x, t1)], ['log'], [x**2], labeled=True)
    assert is_deriv_k(P(x, t1), P(1, t1), DE) == \
        ([(t1, Rational(1, 2))], t1/2, 1)

    DE = DE_of([P(1, x), P(2/(1 + x), t0)], ['log'], [x**2 + 2*x + 1], labeled=True)
    assert is_deriv_k(P(1 + x, t0), P(1, t0), DE) == \
        ([(t0, Rational(1, 2))], t0/2, 1)

    # Issue 10798
    # DE = DifferentialExtension(log(1/x), x)
    DE = DE_of([P(1, x), P(-1/x, t)], ['log'], [1/x], labeled=True)
    assert is_deriv_k(P(1, t), P(x, t), DE) == ([(t, 1)], t, 1)

    # Linearly dependent tower generators (log(x) and log(x**2)) make the
    # structure system underdetermined; the solution must still be a
    # correct full-length coefficient vector (the reduced system used to
    # be misread as a solution vector and silently zip-truncated)
    DE = DE_of([P(1, x), P(1/x, t0), P(2/x, t1)], ['log', 'log'], [x, x**2], labeled=True)
    assert is_deriv_k(P(x, t0), P(1, t0), DE) == \
        ([(t0, 1), (t1, 0)], t0, 1)
    DE = DE_of([P(1, x), P(2/x, t0), P(1/x, t1)], ['log', 'log'], [x**2, x], labeled=True)
    assert is_deriv_k(P(x, t0), P(1, t0), DE) == \
        ([(t0, Rational(1, 2)), (t1, 0)], t0/2, 1)

    # A constant is the derivative of 0
    DE = DE_of([P(1, x)], [], [], labeled=True)
    assert is_deriv_k(P(2, x), P(1, x), DE) == ([], 0, 2)

    # Real towers (Corollary 9.3.2 (i)): hypertangent and arc-tangent
    # monomials never contribute.  t0 == exp(x), t1 == tan(x),
    # t2 == atan(x); log(exp(x)) == x
    DE = DE_of([P(1, x), P(t0, t0), P(1 + t1**2, t1), P(1/(x**2 + 1), t2)],
        ['exp', 'tan', 'atan'], [x, x, x], labeled=True)
    assert is_deriv_k(P(t0, t2), P(1, t2), DE) == ([(x, 1)], x, 1)
    # ... but log(x) and log(tan(x)) are new monomials over it
    assert is_deriv_k(P(x, t2), P(1, t2), DE) is None
    assert is_deriv_k(P(t1, t2), P(1, t2), DE) is None


def test_is_deriv_k_atan() -> None:
    # atan(x) is a monomial over QQ(x) (Example 1 of Bronstein's 1989
    # paper, equation (4) with an empty sum)
    DE = DE_of([P(1, x)], [], [], labeled=True)
    assert is_deriv_k_atan(P(x, x), P(1, x), DE) is None

    # t == atan(x): atan(x) == t and atan(2*x/(1 - x**2)) == 2*t, both
    # up to a constant
    DE = DE_of([P(1, x), P(1/(x**2 + 1), t)], ['atan'], [x], labeled=True)
    assert is_deriv_k_atan(P(x, t), P(1, t), DE) == ([(t, 1)], t)
    assert is_deriv_k_atan(P(2*x, t), P(1 - x**2, t), DE) == \
        ([(t, 2)], 2*t)
    assert is_deriv_k_atan(P(x**2, t), P(1, t), DE) is None

    # t == tan(x): atan(tan(x)) == x up to a constant, and atan(x) is a
    # monomial over QQ(x, tan(x))
    DE = DE_of([P(1, x), P(1 + t**2, t)], ['tan'], [x], labeled=True)
    assert is_deriv_k_atan(P(t, t), P(1, t), DE) == ([(x, 1)], x)
    assert is_deriv_k_atan(P(x, t), P(1, t), DE) is None

    # Exponentials and logarithms never contribute: atan(exp(x)) is a
    # monomial over QQ(x, exp(x))
    DE = DE_of([P(1, x), P(t, t)], ['exp'], [x], labeled=True)
    assert is_deriv_k_atan(P(t, t), P(1, t), DE) is None

    # Both kinds of monomials at once: t1 == atan(x), t2 == tan(x)
    # (the 1989 paper's Example 2): atan of the tangent addition
    # formula for tan(x + 2 - 3*atan(x)/4)
    DE = DE_of([P(1, x), P(1/(x**2 + 1), t1), P(1 + t2**2, t2)], ['atan', 'tan'], [x, x], labeled=True)
    assert is_deriv_k_atan(P(t2 + x, t2), P(1 - t2*x, t2), DE) == \
        ([(x, 1), (t1, 1)], t1 + x)


def test_is_log_deriv_k_t_radical_tan() -> None:
    # tan(x) is a monomial over QQ(x); tan(1) is a constant
    DE = DE_of([P(1, x)], [], [], labeled=True)
    assert is_log_deriv_k_t_radical_tan(P(x, x), P(1, x), DE) is None
    assert is_log_deriv_k_t_radical_tan(P(1, x), P(1, x), DE) == \
        ([], 0, 1, 1)

    # Example 1 of Bronstein's 1989 paper: t == atan(x), and
    # tan(atan(x)/3) is algebraic of degree 3 over QQ(x, atan(x))
    DE = DE_of([P(1, x), P(1/(x**2 + 1), t)], ['atan'], [x], labeled=True)
    assert is_log_deriv_k_t_radical_tan(P(t, t), P(3, t), DE) == \
        ([(t, 1)], t, 3, 0)
    # tan(2*atan(x)) == 2*x/(1 - x**2) is in the field (n == 1)
    assert is_log_deriv_k_t_radical_tan(P(2*t, t), P(1, t), DE) == \
        ([(t, 2)], 2*t, 1, 0)
    # tan(x) and tan(x*atan(x)) are monomials over QQ(x, atan(x))
    assert is_log_deriv_k_t_radical_tan(P(x, t), P(1, t), DE) is None
    assert is_log_deriv_k_t_radical_tan(P(x*t, t), P(1, t), DE) is None

    # Example 2 of the paper: t1 == atan(x), t2 == tan(x), and
    # tan(x + 2 - 3*atan(x)/4) is algebraic of degree 4 over
    # QQ(x, atan(x), tan(x), tan(2))
    DE = DE_of([P(1, x), P(1/(x**2 + 1), t1), P(1 + t2**2, t2)], ['atan', 'tan'], [x, x], labeled=True)
    assert is_log_deriv_k_t_radical_tan(P(x + 2 - Rational(3, 4)*t1, t2),
        P(1, t2), DE) == ([(x, 4), (t1, -3)], 4*x - 3*t1, 4, 2)

    # t == tan(x): tan(2*x + 1) is in QQ(x, tan(x), tan(1)) (n == 1)
    DE = DE_of([P(1, x), P(1 + t**2, t)], ['tan'], [x], labeled=True)
    assert is_log_deriv_k_t_radical_tan(P(2*x + 1, t), P(1, t), DE) == \
        ([(x, 2)], 2*x, 1, 1)
    # but tan(x/2) is algebraic of degree 2 over it
    assert is_log_deriv_k_t_radical_tan(P(x, t), P(2, t), DE) == \
        ([(x, 1)], x, 2, 0)

    # Exponentials and logarithms never contribute: tan(log(x)) is a
    # monomial over QQ(x, log(x))
    DE = DE_of([P(1, x), P(1/x, t)], ['log'], [x], labeled=True)
    assert is_log_deriv_k_t_radical_tan(P(t, t), P(1, t), DE) is None


def test_is_log_deriv_k_t_radical_in_field() -> None:
    # NOTE: any potential constant factor in the second element of the result
    # doesn't matter, because it cancels in Da/a.
    DE = DE_of([P(1, x), P(1/x, t)])
    assert is_log_deriv_k_t_radical_in_field(P(5*t + 1, t), P(2*t*x, t), DE) == \
        (2, R(t*x**5))
    assert is_log_deriv_k_t_radical_in_field(P(2 + 3*t, t), P(5*x*t, t), DE) == \
        (5, R(x**3*t**2))

    DE = DE_of([P(1, x), P(-t/x**2, t)])
    assert is_log_deriv_k_t_radical_in_field(P(-(1 + 2*t), t),
    P(2*x**2 + 2*x**2*t, t), DE) == \
        (2, R(t + t**2))
    assert is_log_deriv_k_t_radical_in_field(P(-1, t), P(x**2, t), DE) == \
        (1, R(t))
    assert is_log_deriv_k_t_radical_in_field(P(1, t), P(2*x**2, t), DE) == \
        (2, R(1/t))

    # exp case: u must come back as an element of the field
    DE = DE_of([P(1, x), P(t, t)])
    assert is_log_deriv_k_t_radical_in_field(P(1, t), P(1, t), DE) == \
        (1, R(t))
    # f == Dt/t + Dt/(t + 1): the log-derivative of t*(t + 1)
    assert is_log_deriv_k_t_radical_in_field(P(2*t + 1, t),
        P(t + 1, t), DE) == (1, R(t**2 + t))

    # Hypertangent case (Section 5.12), t == tan(x): after the residue
    # reduction p == a + b*t, and u == v*(t**2 + 1)**e with n*a == Dv/v
    # and e == n*b/(2*eta), eta == Dt/(t**2 + 1)
    DE = DE_of([P(1, x), P(1 + t**2, t)])
    # D(t**2 + 1)/(t**2 + 1) == 2*t
    assert is_log_deriv_k_t_radical_in_field(P(2*t, t), P(1, t), DE) == \
        (1, R(t**2 + 1))
    assert is_log_deriv_k_t_radical_in_field(P(t, t), P(1, t), DE) == \
        (2, R(t**2 + 1))
    # a != 0 recurses into k: D(x*(t**2 + 1))/(x*(t**2 + 1)), and the
    # same with both n's equal to 2, i.e. (1/2)*D(x*(t**2 + 1))/(...)
    assert is_log_deriv_k_t_radical_in_field(P(1 + 2*t*x, t), P(x, t),
        DE) == (1, R(t**2*x + x))
    assert is_log_deriv_k_t_radical_in_field(P(1 + 2*t*x, t),
        P(2*x, t), DE) == (2, R(t**2*x + x))
    # with a residue term: D(t - 1)/(t - 1) and D((t - 1)*(t**2 + 1))/...
    assert is_log_deriv_k_t_radical_in_field(P(1 + t**2, t), P(t - 1, t),
        DE) == (1, R(t - 1))
    assert is_log_deriv_k_t_radical_in_field(P(1 + t**2 + 2*t*(t - 1), t),
        P(t - 1, t), DE) == (1, R(t**3 - t**2 + t - 1))
    # 1 == Dv/v needs v == exp(x), and x*t needs e == n*x/2, neither in k(t)
    assert is_log_deriv_k_t_radical_in_field(P(1, t), P(1, t), DE) is None
    assert is_log_deriv_k_t_radical_in_field(P(x*t, t), P(1, t), DE) is None
    # the book's precondition sqrt(-1) not in k(t)
    raises(NotImplementedError, lambda: is_log_deriv_k_t_radical_in_field(
        P(I*t, t), P(1, t), DE))
    # nested hypertangents, t1 == tan(x), t2 == tan(tan(x)): the
    # recursion goes through the hypertangent case of the level below
    DE = DE_of([P(1, x), P(1 + t1**2, t1), P((1 + t1**2)*(1 + t2**2), t2)])
    assert is_log_deriv_k_t_radical_in_field(P(2*t1 + 2*(1 + t1**2)*t2, t2),
        P(1, t2), DE) == (1, R(t1**2*t2**2 + t1**2 + t2**2 + 1))


def test_parametric_log_deriv_structure() -> None:
    # The heuristic fails on all of these (z in k at a primitive level);
    # the structure-theorem method (equation (7.44)) decides them.
    DE = DE_of([P(1, x), P(1/x, t)], ['log'], [x], labeled=True)
    # f == 1/x == Dx/x, w == 1: 1*f == Dx/x + 0*w
    assert parametric_log_deriv_structure(P(1, t), P(x, t),
        P(1, t), P(1, t), DE) == (1, 0, R(x))
    # f == 1 + 1/(2*x), w == 1: 2*f == Dx/x + 2*w  (through the wrapper,
    # exercising the heuristic -> structure fallback)
    assert parametric_log_deriv(P(2*x + 1, t), P(2*x, t),
        P(1, t), P(1, t), DE) == (2, 2, R(x))
    # f == 1/(x + 1), w == 1 has the solution (1, 0, x + 1), but x + 1 is
    # not in the tower, so the structure method is inconclusive (None) and
    # the wrapper must raise rather than claim no solution exists.
    assert parametric_log_deriv_structure(P(1, t), P(x + 1, t),
        P(1, t), P(1, t), DE) is None
    raises(NotImplementedError, lambda: parametric_log_deriv(
        P(1, t), P(x + 1, t), P(1, t), P(1, t), DE))

    # Nontrivial n and m: f == 1/(2*x) + 3, w == 2:
    # 2*f == Dx/x + 3*w
    assert parametric_log_deriv_structure(P(6*x + 1, t), P(2*x, t),
        P(2, t), P(1, t), DE) == (2, 3, R(x))

    # Underdetermined system (w lies in the span of the generators) and a
    # w == 0 query.  Which solution is returned depends on the free
    # parameter choice, so check the defining equation n*f == Dv/v + m*w
    # instead of an exact tuple.
    DE = DE_of([P(1, x), P(1/x, t), P(t1, t1)], ['log', 'exp'], [x, x], labeled=True)
    for f, w in [(1/x + 2, R(3)), (1/x + 5, R(0))]:
        fa, fd = frac_in(f, t1)
        wa, wd = frac_in(w, t1)
        A = parametric_log_deriv_structure(fa, fd, wa, wd, DE)
        assert A is not None
        nn, m, v = A
        assert nn > 0
        vv = v.as_expr().subs(t1, exp(x))  # t1 == exp(x) in this extension
        lhs = (R(f)*nn - w*m).as_expr()
        assert (lhs - vv.diff(x)/vv).simplify() == 0


def test_is_deriv_in_field() -> None:
    DE = DE_of([P(1, x)])
    assert is_deriv_in_field(P(2*x, x), P(1, x), DE) == \
        (P(x**2, x), P(1, x))
    assert is_deriv_in_field(P(1, x), P(x, x), DE) is None
    # t == log(x): D(x*t) == t + 1; 1/(x*t) == D(log(log(x))) is not in the field
    DE = DE_of([P(1, x), P(1/x, t)])
    A = is_deriv_in_field(P(t + 1, t), P(1, t), DE)
    assert A is not None
    vp = P(fr(A), t)
    assert derivation(vp, DE) == P(t + 1, t)
    assert is_deriv_in_field(P(1, t), P(x*t, t), DE) is None
    # t == exp(x): D(x*t**2) == (2*x + 1)*t**2; t is not a derivative
    DE = DE_of([P(1, x), P(t, t)])
    A = is_deriv_in_field(P((2*x + 1)*t**2, t), P(1, t), DE)
    assert A is not None
    vp = P(fr(A), t)
    assert derivation(vp, DE) == P((2*x + 1)*t**2, t)


def test_parametric_log_deriv() -> None:
    DE = DE_of([P(1, x), P(1/x, t)])
    assert parametric_log_deriv_heu(P(5*t**2 + t - 6, t), P(2*x*t**2, t),
    P(-1, t), P(x*t**2, t), DE) == \
        (2, 6, R(t*x**5))

    DE = DE_of([P(1, x)])
    assert parametric_log_deriv_heu(P(-1, x), P(1, x), P(1, x),
    P(1, x), DE) ==\
        (1, -1, R(1))

    assert parametric_log_deriv_heu(P(-2, x), P(5, x), P(2, x),
    P(3, x), DE) ==\
        (5, -3, R(1))

    # f == w == 0: any n works, with m == 0 and v == 1 (the rational
    # shortcut used to raise ZeroDivisionError on this input, which
    # arises from the degenerate hypertangent case in test_special_denom())
    assert parametric_log_deriv_heu(P(0, x), P(1, x), P(0, x),
    P(1, x), DE) == \
        (1, 0, R(1))

    # Cases where z is in k, so the residue equations are vacuous.  These
    # used to return None ("proven no solution"), which was wrong: at the
    # base level they are completely decidable from the polynomial parts.
    # f == -1 + 1/(x + 1), w == 1
    assert parametric_log_deriv_heu(P(-x, x), P(x + 1, x), P(1, x),
    P(1, x), DE) == \
        (1, -1, R(x + 1))
    # f == 1/x, w == 1 (arises while integrating log(log(x + exp(x))))
    assert parametric_log_deriv_heu(P(1, x), P(x, x), P(1, x),
    P(1, x), DE) == \
        (1, 0, R(x))
    # f == 1/(x**2 + 1), w == 1: proven no solution (complex residues)
    assert parametric_log_deriv_heu(P(1, x), P(x**2 + 1, x), P(1, x),
    P(1, x), DE) is None

    # In a non-base field the z in k case cannot be decided by the
    # heuristic ("failed" in the book), so it must raise
    # NotImplementedError, not return None.
    DE = DE_of([P(1, x), P(1/x, t)])
    raises(NotImplementedError, lambda: parametric_log_deriv_heu(
        P(1, t), P(t + 1, t), P(1, t), P(1, t), DE))
