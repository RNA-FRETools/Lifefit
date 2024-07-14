from pathlib import Path
import importlib.metadata

DATA_DIR = Path(__file__).resolve().parents[2].joinpath("data")
TEST_DIR = Path(__file__).resolve().parents[2].joinpath("tests")
VERSION = importlib.metadata.version("lifefit")

from . import tcspc
