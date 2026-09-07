"""Helpers for the test suite."""
from __future__ import annotations

from typing import Callable

__all__ = ['untyped']


def untyped(f: Callable[..., object]) -> Callable[..., object]:
    """``f`` with unchecked argument types.

    The tests which check that wrong argument types are rejected at run
    time call the function under test through this helper, so that the
    wrong types are not also a static type error.
    """
    return f
