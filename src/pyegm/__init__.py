"""
PyEGM - Explosion Generation Model

A scikit-learn compatible implementation of the EGM algorithm for classification tasks.
Core features include hypersphere/gaussian point generation and adaptive radius adjustment.
"""
__version__ = "0.2.0"
__author__ = "CJiangqiu"
__author_email__ = "17787153839@qq.com"
__license__ = "MIT"

from .pyegm import PyEGM

__all__ = ['PyEGM']

import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

import sys
if sys.version_info < (3, 7):
    raise ImportError(
        f"PyEGM requires Python 3.7 or later. "
        f"Current Python version: {sys.version}"
    )