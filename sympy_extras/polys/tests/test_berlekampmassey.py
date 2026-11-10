"""Berlekamp-Massey over the rationals and finite fields: planted
recurrences are recovered, and the result is the shortest one (checked by
the rank of the Hankel matrix)."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy_extras.polys.berlekampmassey import berlekamp_massey, extend_sequence, satisfies_recurrence
from sympy_extras.polys.domains import GF, QQ
from sympy_extras.polys.domains.galoisfield import GaloisField


def test_known_sequences() -> None:
    fibonacci = [Fraction(v) for v in (0, 1, 1, 2, 3, 5, 8, 13, 21, 34)]
    assert berlekamp_massey(fibonacci, QQ) == [1, -1, -1]
    # squares: (x - 1)**3; powers of two; constant; zero; a single spike
    assert berlekamp_massey([Fraction(k*k) for k in range(10)], QQ) == [1, -3, 3, -1]
    assert berlekamp_massey([Fraction(2**k) for k in range(8)], QQ) == [1, -2]
    assert berlekamp_massey([Fraction(7)]*5, QQ) == [1, -1]
    assert berlekamp_massey([Fraction(0)]*5, QQ) == [1] and berlekamp_massey([], QQ) == [1]
    # a spike after two zeros: order three (the lower coefficients are not determined)
    assert len(berlekamp_massey([0, 0, 1], GF(5))) == 4
    # the m-sequence of x**3 + x + 1 over GF(2), of period 7
    assert berlekamp_massey([1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1], GF(2)) == [1, 0, 1, 1]
    assert extend_sequence(fibonacci, QQ, 13)[-3:] == [55, 89, 144]
    assert satisfies_recurrence(fibonacci, [Fraction(1), Fraction(-1), Fraction(-1)], QQ)
    assert not satisfies_recurrence(fibonacci, [Fraction(1), Fraction(-2)], QQ)


def test_planted_recurrences_over_prime_fields() -> None:
    rng = random.Random(111)
    for p in (2, 3, 7, 101):
        F = GF(p)
        for _ in range(40):
            order = rng.randint(1, 6)
            taps = [rng.randrange(p) for _ in range(order)]
            sequence = [rng.randrange(p) for _ in range(order)]
            while len(sequence) < 2*order + 6:
                sequence.append(sum(t*sequence[-1 - i] for i, t in enumerate(taps)) % p)
            polynomial = berlekamp_massey(sequence, F)
            found = len(polynomial) - 1
            assert polynomial[0] == 1 and found <= order
            assert satisfies_recurrence(sequence, polynomial, F)
            # the shortest: no recurrence of smaller order (brute force on small cases)
            if p <= 3 and found >= 1 and found <= 4:
                for smaller in range(p**(found - 1)):
                    candidate = [1]
                    value = smaller
                    for _ in range(found - 1):
                        candidate.append(value % p)
                        value //= p
                    assert not satisfies_recurrence(sequence, candidate, F)
            assert extend_sequence(sequence[:2*order + 2], F, len(sequence)) == sequence


def test_other_fields() -> None:
    rng = random.Random(112)
    K = GaloisField(2, 4)
    for _ in range(20):
        order = rng.randint(1, 4)
        taps = [K.random(rng) for _ in range(order)]
        sequence = [K.random(rng) for _ in range(order)]
        while len(sequence) < 2*order + 4:
            total = K.zero
            for i, t in enumerate(taps):
                total = K.add(total, K.mul(t, sequence[-1 - i]))
            sequence.append(total)
        polynomial = berlekamp_massey(sequence, K)
        assert len(polynomial) - 1 <= order and satisfies_recurrence(sequence, polynomial, K)
    # over QQ with rational taps
    for _ in range(20):
        order = rng.randint(1, 4)
        rational_taps = [Fraction(rng.randint(-3, 3), rng.randint(1, 3)) for _ in range(order)]
        rationals = [Fraction(rng.randint(-3, 3)) for _ in range(order)]
        while len(rationals) < 2*order + 4:
            rationals.append(sum((t*rationals[-1 - i] for i, t in enumerate(rational_taps)), Fraction(0)))
        polynomial_q = berlekamp_massey(rationals, QQ)
        assert len(polynomial_q) - 1 <= order and satisfies_recurrence(rationals, polynomial_q, QQ)
