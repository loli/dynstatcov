=====================
How to use DynStatCov
=====================

DynStatCov comes with a simple interface for the following work-flow:

1. Compute an initial co-variance matrix from a base set of observations
2. Add a new observation to update the co-variance matrix
3. Get the current co-variance matrix as numpy array
4. Go to step 2.

See the library reference for a description of the complete functionality.

Examples
--------
Example with initially three observations of four features each, which is
updated once.

>>> import numpy as np
>>> from dynstatcov import Dynstatcov
>>> X = np.array([[3, 4, 1, 2],
[4, 4, 0, 3],
[1, 3, 4, 2]], dtype=numpy.float64)

>>> dsc = Dynstatcov(X)
>>> dsc.get_cov()
array([[ 2.33333397, 0.83333397, -3.16666698, 0.66666603],
[ 0.83333397, 0.33333206, -1.16666603, 0.16666603],
[-3.16666698, -1.16666603, 4.33333302, -0.83333397],
[ 0.66666603, 0.16666603, -0.83333397, 0.33333397]], dtype=float64)

>>> dsc.get_n_samples()
3

>>> y = np.array([3, 4, 1, 2], dtype=numpy.float64)
>>> dsc.update(y)
>>> dsc.get_cov()
array([[ 1.58333337, 0.58333337, -2.16666675, 0.41666669],
[ 0.58333337, 0.25 , -0.83333337, 0.08333334],
[-2.16666675, -0.83333337, 3. , -0.5 ],
[ 0.41666669, 0.08333334, -0.5 , 0.25 ]], dtype=float64)

Internally, only the upper part of the symmetrical co-variance matrix is
stored and the full matrix reconstructed on each call to *get_cov()*.
Alternatively, you can also request the upper triangular matrix, which is
slightly faster.

>>> dsc.get_cov_tri()
array([ 1.58333337, 0.58333337, -2.16666675, 0.41666669, 0.25 ,
-0.83333337, 0.08333334, 3. , -0.5 , 0.25 ], dtype=float64)

We can also remove a sample setting the optional second argument of `update()` non-zero.

>>> dsc.update(y, 1)
>>> dsc.get_cov()
array([[ 2.33333397,  0.83333397, -3.16666698,  0.66666603],
       [ 0.83333397,  0.33333206, -1.16666603,  0.16666603],
       [-3.16666698, -1.16666603,  4.33333302, -0.83333397],
       [ 0.66666603,  0.16666603, -0.83333397,  0.33333397]], dtype=float64)

Note on precision
-----------------
DynStatCov can be compiled with different precisions (e.g *numpy.float32* or *numpy.float64*).
You might have to test, which version you are running. By default, its double precision.
If required, you can always change typedef of `DTYPE_t` in the cython code and re-compile, as described in :doc:`../installation/configuration`.

Note on update() method
-----------------------
The sample will not get checked for its length. You are yourself
responsible, that it fits the length of the observations passed
on class initialization.

