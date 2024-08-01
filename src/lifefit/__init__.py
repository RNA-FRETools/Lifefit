from pathlib import Path
import importlib.metadata

DATA_DIR = Path(__file__).resolve().parent.joinpath("data")
APP_DIR = Path(__file__).resolve().parent.joinpath("app")
VERSION = importlib.metadata.version("lifefit")

from . import tcspc
