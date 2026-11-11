"""Differential geometry on the typed core: the curvature of a metric
given by its components (:mod:`sympy_extras.diffgeom.metric`), curves and
surfaces in space (:mod:`sympy_extras.diffgeom.classical`)."""
from .classical import Curve, Surface
from .metric import Metric

__all__ = ['Metric', 'Curve', 'Surface']
