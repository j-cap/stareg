.. stareg documentation master file, created by
   sphinx-quickstart on Wed Jul 29 18:11:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to stareg's documentation!
==================================

Introduction
------------

**ST**\ ructured **A**\ dditive **REG**\ ression is a semi-parameteric regression approach of the form:

.. math::
   \hat y = f(X) = f_1(x_1) + f_2(x_2) + f_3(x_1, x_2) + ... + f_d(x_d)

where the :math:`x_i` are independet variables and :math:`\hat y` is the prediction for the dependent variable :math:`y`.

The functions :math:`f_i` are generated using P-splines, short for penalized spline. The additive approach allows
to model multi-dimensional non-linear relationships is a straightforward manner. 

This results in a flexible model, for which the incorporation of prior knowledge and the controlling of overfitting is 
easily possible. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   rst/overview
   rst/usage
   rst/stareg
   rst/authors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



UPDATE TEST, commit am 30.07.2020 17:59