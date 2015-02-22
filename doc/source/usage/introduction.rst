===================
What is DynStatCov?
===================

DynStatCov is a Cython Library for fast dynamic statistical co-variance update. It is intended for usage in applications, where a statistical co-variance matrix has to be computed from observations and periodically updated. The naive approach would require to re-compute the matrix every time from all samples and to hold them in memory. By re-organizing the formula (see :doc:`background` for details), the computation can be realized using intermediate sums and thus achieving a higher speed.

Complexity
----------
The update is considerably cheaper than the computation of the complete sample co-variance matrix, not only when :math:`n>>m`:

- **complete:** :math:`\mathcal O(nm^2)`
- **update:** :math:`\mathcal O(m^2)`

Speed
-----
Tested with *%timeit* using 10000 initial observations of 3 features each and then computing an update upon arrival of a new observation:

- complete re-computation with *numpy.cov*: **553us**
- dynamic update implemented in Python: **66us**
- DynStatCov dynamic update: **0.804us**


