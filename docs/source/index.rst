.. LifeFit documentation master file, created by
   sphinx-quickstart on Sat Apr 13 15:33:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LifeFit Documentation
=====================

.. image:: https://github.com/fdsteffen/LifeFit/workflows/LifeFit%20build/badge.svg
  :target: https://github.com/fdsteffen/LifeFit/actions

.. image:: https://readthedocs.org/projects/lifefit/badge/?version=latest
  :target: https://lifefit.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/pypi/v/lifefit
  :target: https://pypi.org/project/lifefit/

.. image:: https://anaconda.org/fdsteffen/lifefit/badges/installer/conda.svg
  :target: https://anaconda.org/fdsteffen/lifefit

What is LifeFit
---------------

LifeFit is a Python package to analyze **time-correlated single-photon counting (TCSPC)** data sets, namely **fluorescence lifetime** and **time-resolve anisotropy** decays.

Webserver
---------

You can run LifeFit directly in your browser: https://tcspc-lifefit.herokuapp.com/

.. image:: _static/webserver.png

.. note::

  Initial startup of the webserver might take a few seconds, please be patient.


Installation
------------

There are different options how to install LifeFit. 

Conda
*****

Install the package into your conda environment ::

    conda install -c fdsteffen lifefit

PyPI
****

Alternatively, you can install the latest release of with pip ::

    pip install lifefit

Install from source
*******************

Finally, you can also get the latest development version directly from Github ::

    pip install git+https://github.com/fdsteffen/Lifefit.git


Tutorial
--------

For an introduction into the functionality of LifeFit visit the :doc:`tutorial <tutorial/lifefit_tutorial>`. The Jupyter Notebook can be downloaded :download:`here <tutorial/lifefit_tutorial.ipynb>`.

Bug reports
-----------

Please report any *bugs* via the `issue tracker <https://github.com/fdsteffen/Lifefit/issues>`_ on Github.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:
   
   tutorial/lifefit_tutorial
   lifefit


Reference
---------

To cite LifeFit, please refer to the following paper:

F.D. Steffen, R.K.O. Sigel, R. Börner, *Phys. Chem. Chem. Phys.* **2016**, *18*, 29045-29055. [![](https://img.shields.io/badge/DOI-10.1039/C6CP04277E-blue.svg)](https://doi.org/10.1039/C6CP04277E)

