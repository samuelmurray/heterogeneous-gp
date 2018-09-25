"""
Module for all supported models
"""

__all__ = [
    "GP",
    "GPLVM",
    "MLGP",
    "MLGPLVM",
]

from .gp import GP
from .gplvm import GPLVM
from .mlgp import MLGP
from .mlgplvm import MLGPLVM
