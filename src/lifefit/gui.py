from streamlit.web import cli
from lifefit import APP_DIR


def main():
    """Run streamlit GUI of LifeFit"""
    cli.main_run([str(APP_DIR.joinpath("app.py"))])
