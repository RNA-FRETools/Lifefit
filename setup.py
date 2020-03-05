import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lifefit",
    version="1.0.0",
    author="Fabio Steffen",
    description="Fitting TCSPC data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fdsteffen/lifefit",
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='TCSPC, fluorescence, lifetime, anisotropy'
)
