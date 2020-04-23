.. LifeFit documentation master file, created by
   sphinx-quickstart on Sat Apr 13 15:33:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LifeFit Documentation
=====================

What is LifeFit
---------------

LifeFit is a Python package to analyze **time-correlated single-photon counting (TCSPC)** data sets, namely **fluorescence lifetime** and **time-resolve anisotropy** decays.

Webserver
---------

You can run LifeFit directly in your browser: https://tcspc-lifefit.herokuapp.com/

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

Dependencies
------------

Lifefit depends on the following Python packages:

- numpy
- scipy
- uncertainties 


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

