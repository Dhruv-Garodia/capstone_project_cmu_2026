"""Éponge: segmentation and microstructure analysis for pFIB-SEM stacks of porous catalyst layers."""

__version__ = "0.2.0"

from . import metrics  # noqa: F401
from .pipeline import Result, analyze_file, analyze_image, analyze_stack  # noqa: F401
