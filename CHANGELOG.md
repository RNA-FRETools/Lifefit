## Changelog
All notable changes to this project are documented in this file.

### [1.0.10] 2025-02-19
#### Changed
- Fix integer casting of intensity counts
- Improve error catching in Horiba reader function

### [1.0.9] 2024-11-3
#### Added
- Update Horiba reader function
- Update doc strings and test

### [1.0.8] 2024-08-1
#### Added
- Relocate `data` directory

### [1.0.7] 2024-08-1
#### Added
- Add `lifefit_gui` CLI to start streamlit app

### [1.0.6] 2024-08-1
#### Added
- Add reader to parse intensity counts along a time axis instead of channels.
- Update tutorial and streamlit app to use `fileformat="time_intensity"`

### [1.0.5] 2024-07-14
#### Added
- Refactor to use pyproject.toml
- Add streamlit to run app locally
- Refactor docs to use mkdocs

### [1.0.4] 2021-02-08
#### Added
- Add source of Streamlit web app
- Add download link for tabular fit data
- Add manual range selection for anisotropy fit in web app

#### Changed
- Turn fits and residuals into integer counts
- Use pytest with coverage report to execute unittest suite

### [1.0.3] 2020-04-23
#### Added
- Add json export of data and fit results

### [1.0.2] 2020-04-15
#### Added
- Badges for anaconda and PyPi

#### Changed
- Installation instructions for conda and pip in README.md and Docs

### [1.0.1] 2020-04-15
#### Added
- Main module tcspc.py
- Tests for Horiba import, reconvolution fit (with experimental and Gaussian IRF)
- Docs on [ReadTheDocs](https://lifefit.readthedocs.io/en/latest/)
- CI with [Github Actions](https://github.com/fdsteffen/Lifefit/actions)
