"""Functions of the typed core (the counterpart of :mod:`sympy.functions`)."""
from .elementary.exponential import Exp, Log, exp, log
from .elementary.hyperbolic import Cosh, Sinh, Tanh, cosh, sinh, tanh
from .elementary.trigonometric import Cos, Sin, Tan, cos, sin, tan

__all__ = ['Exp', 'Log', 'exp', 'log', 'Sin', 'Cos', 'Tan', 'sin', 'cos', 'tan', 'Sinh', 'Cosh', 'Tanh',
           'sinh', 'cosh', 'tanh']
