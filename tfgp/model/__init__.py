"""
Package for all supported models
"""

__all__ = [
    "BatchMLGP",
    "GP",
    "GPLVM",
    "MLGP",
    "MLGPLVM",
]

from .batch_mlgp import BatchMLGP
from .gp import GP
from .gplvm import GPLVM
from .mlgp import MLGP
from .mlgplvm import MLGPLVM
