"""
Package for all supported models
"""

__all__ = [
    "BatchMLGP",
    "BatchMLGPLVM",
    "GP",
    "GPLVM",
    "InducingPointsModel",
    "MLGP",
    "MLGPLVM",
    "Model"
]

from .batch_mlgp import BatchMLGP
from .batch_mlgplvm import BatchMLGPLVM
from .gp import GP
from .gplvm import GPLVM
from .inducing_points_model import InducingPointsModel
from .mlgp import MLGP
from .mlgplvm import MLGPLVM
from .model import Model
