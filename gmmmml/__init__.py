
import logging

__version__ = "0.0.1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

del handler, logger, logging

from . import (em, mml, gmm)
from .gmm import GaussianMixture