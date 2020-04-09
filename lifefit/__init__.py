#from lifefit.tcspc import *
from . import tcspc
import os

_SRC_DIR = os.path.abspath(os.path.dirname(__file__))
_DOCS_DIR = os.path.join(os.path.dirname(_SRC_DIR), 'docs')
_DATA_DIR = os.path.join(os.path.dirname(_SRC_DIR), 'data')
_TEST_DIR = os.path.join(os.path.dirname(_SRC_DIR), 'tests')
