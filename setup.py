import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

about = {}
with open('lifefit/__about__.py') as a:
    exec(a.read(), about)

INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'uncertainties'
    ]

setuptools.setup(
    name=about['__title__'],
    version=about['__version__'],
    author=about['__author__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about['__url__'],
    project_urls=about['__project_urls__'],
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    classifiers=about['__classifiers__'],
    keywords=about['__keywords__']
)
