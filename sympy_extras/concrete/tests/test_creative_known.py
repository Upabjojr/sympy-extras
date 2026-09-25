from __future__ import annotations

from sympy import Integer, binomial, harmonic, simplify, symbols

from sympy_extras.concrete import definite_sum

n, k = symbols('n k')


def test_paule_schneider_harmonic_cubes() -> None:
    # P. Paule, C. Schneider, Computer proofs of a new family of harmonic
    # number identities, Adv. Appl. Math. 31 (2003): the case of the
    # third powers, sum (1 + 3 (n - 2k) H_k) binomial(n, k)**3 = (-1)**n
    f = (1 + 3*(n - 2*k)*harmonic(k))*binomial(n, k)**3
    result = definite_sum(f, (k, 0, n))
    assert result is not None and simplify(result - (-1)**n) == 0
    for m in range(0, 7):
        direct = sum((f.subs({n: m, k: i}) for i in range(0, m + 1)), Integer(0))
        assert direct == (-1)**m
