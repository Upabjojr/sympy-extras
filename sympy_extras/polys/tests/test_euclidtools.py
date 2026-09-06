from __future__ import annotations

from sympy.matrices import Matrix
from sympy.polys import ring, ZZ
from sympy.testing.pytest import raises

from sympy_extras.polys.euclidtools import dup_psc, dmp_psc, psc


def test_dup_psc():
    R, x = ring("x", ZZ)

    assert psc(R(0), R(0)) == []
    assert psc(x**2 + 1, 0) == []
    assert psc(0, x**2 + 1) == []
    assert psc(R(3), 5) == [1]
    assert psc(x**2 + 1, x**2 - 1) == [4, 0, 1]
    assert psc(x**2 - 1, x**2 + 1) == [4, 0, 1]
    assert psc(x**3 - x, x**2 - 1) == [0, 0, 1]
    assert psc(x**2 - 1, x**3 - x) == [0, 0, 1]
    assert psc(x**3 - 3*x**2 + 2*x, x**3 - x) == [0, 0, 3, 1]
    assert psc(x**5 + x**3 - 1, 2*x**3) == [32, 0, 0, 4]
    assert psc(x**4 + x + 1, 3*x**2 - 2) == [115, -27, 9]
    raises(TypeError, lambda: psc(0, 0))

    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    assert psc(f, g)[0] == R.dup_resultant(f, g)
    assert psc(f, g)[-1] == 9

    # the dense interface
    assert dup_psc(f.to_dense(), g.to_dense(), ZZ) == psc(f, g)
    assert dup_psc([], [ZZ(1)], ZZ) == []

    # principal subresultant coefficients as determinants of the Sylvester
    # submatrices, checked on a few cases with degree jumps and common roots
    def psc_det(f, g):
        n, m = len(f) - 1, len(g) - 1
        out = []
        for j in range(m + 1):
            rows = []
            width = n + m - j
            for i in range(m - j):
                rows.append([0]*i + f + [0]*(width - n - 1 - i))
            for i in range(n - j):
                rows.append([0]*i + g + [0]*(width - m - 1 - i))
            M = Matrix(rows)
            out.append(M[:, :n + m - 2*j].det() if M.rows else 1)
        return [ZZ(c) for c in out]

    cases = [
        ([1, 0, 1, 0, -1], [2, 0, 0, 0]),
        ([1, -1, 0, 2, 3], [1, 0, 0, -1]),
        ([2, -3, 1, 0, 0, -1, 5], [1, 0, 0, 0, 0, 2]),
        ([1, -2, 1], [1, -1, 0]),
        ([1, 0, -2, 0, 1], [1, 0, -1]),
        ([3, 1, -4, 1, 2, -1], [1, 2, 0, 0, -3]),
    ]
    for f, g in cases:
        assert psc(R.from_list(f), R.from_list(g)) == psc_det(f, g)
        assert dup_psc([ZZ(c) for c in f], [ZZ(c) for c in g], ZZ) == psc_det(f, g)


def test_dmp_psc():
    R, x, y = ring("x,y", ZZ)

    assert psc(R(0), R(0)) == []
    assert psc(x*y + 1, 0) == []
    assert psc(x**2*y + x, x + y) == [(y**3 - y).drop(x), 1]
    assert psc(x**2*y + x, 3) == [9]
    assert psc(x**2 - y, x**2*y - 1) == [(y**4 - 2*y**2 + 1).drop(x), 0, 1]

    f = x**2 + y**2 - 1
    assert psc(f, f.diff(x)) == [(4*y**2 - 4).drop(x), 2]
    assert psc(f, x - y) == [(2*y**2 - 1).drop(x), 1]
    assert psc(f, x - y)[0] == R.dmp_resultant(f, x - y)

    g = (x - y)*(x**2 + 1)
    h = (x - y)*(x + 3)
    assert psc(g, h) == [0, 10, 1]

    # the dense interface
    assert dmp_psc(f.to_dense(), (x - y).to_dense(), 1, ZZ) == [[ZZ(2), ZZ(0), ZZ(-1)], [ZZ(1)]]
    u = [ZZ(1), ZZ(0), ZZ(1)]
    assert dmp_psc(u, [ZZ(1), ZZ(0), ZZ(-1)], 0, ZZ) == dup_psc(u, [ZZ(1), ZZ(0), ZZ(-1)], ZZ)
    assert dmp_psc([[]], f.to_dense(), 1, ZZ) == []

    R, x, y, z = ring("x,y,z", ZZ)
    f = x**2 + y**2 + z**2 - 1
    coeffs = psc(f, x*y - z)
    assert coeffs == [R.dmp_resultant(f, x*y - z), y.drop(x)]
    assert coeffs[0] == (y**4 + y**2*z**2 - y**2 + z**2).drop(x)
