# encoding: utf-8
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False

"""
A run-time optimized library for dynamic updates of a statistical co-variance matrix.

Copyright (C) 2015 Oskar Maier
 
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

##########
# Changelog
# 2015-02-22 added a subtraction option for update()
# 2015-02-20 properly documented and tested
# 2015-02-17 created
##########

# python imports
import numpy as np

# cython imports
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# extern cdefs

# type definitions

# docstring info
__author__ = "Oskar Maier"
__copyright__ = "Copyright 2015, Oskar Maier"
__version__ = "0.1.0"
__maintainer__ = "Oskar Maier"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Development"

cdef class Dynstatcov:
    r"""
    Dynamically updateable statistical co-variance matrix.
    
    Implements a minimal method for dynamic updates of a statistic co-variance
    matrix upon the arrival of a new observation. Optimized for speed and low
    memory requirements: the computational and memory requirements depend
    only on the number of features in each observation vector and not on the
    number of observations.    
    
    Parameters
    ----------
    X : ndarray
        Two-dimensional row-major/C-ordered matrix with samples.    
    
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
    array([[ 2.33333397,  0.83333397, -3.16666698,  0.66666603],
           [ 0.83333397,  0.33333206, -1.16666603,  0.16666603],
           [-3.16666698, -1.16666603,  4.33333302, -0.83333397],
           [ 0.66666603,  0.16666603, -0.83333397,  0.33333397]], dtype=float64)

    >>> dsc.get_n_samples()
    3

    >>> y = np.array([3, 4, 1, 2], dtype=numpy.float64)
    >>> dsc.update(y)
    >>> dsc.get_cov()
    array([[ 1.58333337,  0.58333337, -2.16666675,  0.41666669],
           [ 0.58333337,  0.25      , -0.83333337,  0.08333334],
           [-2.16666675, -0.83333337,  3.        , -0.5       ],
           [ 0.41666669,  0.08333334, -0.5       ,  0.25      ]], dtype=float64)
           
    Internally, only the upper part of the symmetrical co-variance matrix is
    stored and the full matrix reconstructed on each call to `get_cov()`.
    Alternatively, you can also request the upper triangular matrix, which is
    slightly faster.
           
    >>> dsc.get_cov_tri()
    array([ 1.58333337,  0.58333337, -2.16666675,  0.41666669,  0.25      ,
           -0.83333337,  0.08333334,  3.        , -0.5       ,  0.25      ], dtype=float64)
           
    We can also remove a sample setting the optional second argument of `update()` non-zero.
    
    >>> dsc.update(y, 1)
    >>> dsc.get_cov()
    array([[ 2.33333397,  0.83333397, -3.16666698,  0.66666603],
           [ 0.83333397,  0.33333206, -1.16666603,  0.16666603],
           [-3.16666698, -1.16666603,  4.33333302, -0.83333397],
           [ 0.66666603,  0.16666603, -0.83333397,  0.33333397]], dtype=float64)
           
    Notes
    -----
    Dynstatcov can be compiled with either single (`numpy.float32`) or
    double precision (`numpy.float64`). You might have to test, which version
    you are running. By default, its double precision.
    If required, you can always change typedef of `DTYPE_t` in the cython code
    and re-compile.
    """

    def __cinit__(self, DTYPE_t[:,::1] X not None):
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t n_upper = upper_n_elements(n_features)
         
        cdef DTYPE_t* squaresum = <DTYPE_t*> PyMem_Malloc(n_upper * sizeof(DTYPE_t))
        cdef DTYPE_t* sum = <DTYPE_t*> PyMem_Malloc(n_features * sizeof(DTYPE_t))
          
        cdef SIZE_t i = 0
          
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_upper = n_upper
        self.squaresum = squaresum
        self.sum = sum
        self.cov = <DTYPE_t*> PyMem_Malloc(n_upper * sizeof(DTYPE_t))
        self.__mean = <DTYPE_t*> PyMem_Malloc(n_features * sizeof(DTYPE_t))
          
        if not self.cov or not self.squaresum or not self.sum or not self.__mean:
            raise MemoryError()
        
        # initialize with 0s
        fill_zeros(sum, n_features)
        fill_zeros(squaresum, n_upper)
        
        # initialize cov matrix construction elements        
        for i in range(n_samples):
            vector_add(sum, &X[i][0], n_features) # or X.buf + i * n_features ?
            upper_add_sample_autocorrelation_matrix(squaresum, &X[i][0], n_features)
              
        self.__compute_covariance_matrix()
        
    def __dealloc__(self):
        PyMem_Free(self.cov)
        PyMem_Free(self.squaresum)
        PyMem_Free(self.sum)
        PyMem_Free(self.__mean)
        
    cpdef update(self, DTYPE_t[::1] x, int subtract = 0):
        r"""
        Add a new sample and update the co-variance matrix.
        
        Parameters
        ----------
        x : ndarray
            One-dimensional row-major/C-ordered array representing an
            observation.
        subtract : int
            If non-zero, the new sample will be removed rather than added.
            
        Notes
        -----
        The sample will not get checked for its length. You are yourself
        responsible, that it fits the length of the observations passed
        on class initialization.
        """
        if 0 == subtract:
            self.__update_add(&x[0])
        else:
            self.__update_sub(&x[0])
            
    cpdef np.ndarray get_cov(self):
        r"""
        Full co-variance matrix.
        
        Returns
        -------
        cov : ndarry
            The full co-variance matrix as numpy array.
        """
        cdef SIZE_t n_features = self.n_features
        cdef DTYPE_t* cov = self.cov
        
        cdef DTYPE_t [:,::1] cov_full_view = <DTYPE_t[:n_features,:n_features]> PyMem_Malloc(n_features * n_features * sizeof(DTYPE_t))
        
        if not &cov_full_view[0][0]:
            raise MemoryError()
        
        upper_to_matrix(&cov_full_view[0][0], cov, n_features)
        
        cdef np.ndarray arr = np.asarray(cov_full_view).copy()
        
        PyMem_Free(&cov_full_view[0][0])
        
        return arr
    
        # alternative, possibly safer version to create numpy array from data
#         cdef pointer_to_numpy_array_complex128(void * ptr, np.npy_intp size):
#             '''Convert c pointer to numpy array.
#             The memory will be freed as soon as the ndarray is deallocated.
#             '''
#             cdef extern from "numpy/arrayobject.h":
#                 void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
#             cdef np.ndarray[np.complex128, ndim=1] arr = \
#                     np.PyArray_SimpleNewFromData(1, &size, np.NPY_COMPLEX128, ptr)
#             PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
#             return arr    
        
    cpdef np.ndarray get_cov_tri(self):
        r"""
        Upper triangular part of the co-variance matrix.
        
        Returns
        -------
        cov : ndarry
            The upper triangular part of the co-variance matrix as 1-D numpy array.
        """
        cdef SIZE_t n_upper = self.n_upper
        cdef DTYPE_t* cov = self.cov

        cdef DTYPE_t [::1] cov_view = <DTYPE_t[:n_upper]> cov
        return np.asarray(cov_view).copy()
        
    cpdef int get_n_samples(self):
        r"""
        Number of samples.
        
        Returns
        -------
        n_samples : int
            The number of samples used so far to compute the co-variance
            matrix.
        """
        cdef SIZE_t n_samples = self.n_samples
        return n_samples
        
    cdef void __update_add(self, DTYPE_t* x) nogil:
        "Add a new sample and update the co-variance matrix."
        cdef DTYPE_t* sum = self.sum
        cdef DTYPE_t* squaresum = self.squaresum
        cdef SIZE_t n_samples = self.n_samples
        cdef SIZE_t n_features = self.n_features
        
        n_samples += 1
        upper_add_sample_autocorrelation_matrix(squaresum, x, n_features)
        vector_add(sum, x, n_features)
        
        self.n_samples = n_samples
        
        self.__compute_covariance_matrix()
        
    cdef void __update_sub(self, DTYPE_t* x) nogil:
        "Remove a sample and update the co-variance matrix."
        cdef DTYPE_t* sum = self.sum
        cdef DTYPE_t* squaresum = self.squaresum
        cdef SIZE_t n_samples = self.n_samples
        cdef SIZE_t n_features = self.n_features
        
        n_samples -= 1
        upper_sub_sample_autocorrelation_matrix(squaresum, x, n_features)
        vector_sub(sum, x, n_features)
        
        self.n_samples = n_samples
        
        self.__compute_covariance_matrix()        
        
    cdef void __compute_covariance_matrix(self) nogil:
        "Compute the co-variance matrix from its components."
        cdef DTYPE_t* cov = self.cov
        cdef DTYPE_t* sum = self.sum
        cdef DTYPE_t* squaresum = self.squaresum
        cdef SIZE_t n_samples = self.n_samples
        cdef SIZE_t n_features = self.n_features
        cdef SIZE_t n_upper = self.n_upper
        
        cdef DTYPE_t* mean = self.__mean
        
        # create mean of samples sum vector
        fill_zeros(mean, n_features)
        vector_add(mean, sum, n_features)
        vector_multiply_scalar(mean, 1.0/n_samples, n_features)
        
        # compute co-variance matrix
        fill_zeros(cov, n_upper)
        upper_add_sample_autocorrelation_matrix(cov, mean, n_features)
        vector_multiply_scalar(cov, n_samples, n_upper)
        upper_sub_outer_product_eachway(cov, mean, sum, n_features)
        vector_add(cov, squaresum, n_upper)
        vector_multiply_scalar(cov, 1.0/(n_samples - 1), n_upper)
    
cdef inline void upper_to_matrix(DTYPE_t* X, DTYPE_t* Y, SIZE_t length) nogil:
    "Convert the upper triangular matrix Y to full matrix X assuming symmetry."
    cdef SIZE_t p1 = 0
    cdef SIZE_t p2 = 0
    
    # first copy existing elements to upper
    for p1 in range(length):
        for p2 in range(p1, length):
            X[p2 + p1 * length] = Y[0]
            Y += 1
    
    # copy triangular symmetric elements from upper to lower (excluding diagonal)
    for p1 in range(1, length):
        for p2 in range(0, p1):
            X[p2 + p1 * length] = X[p1 + p2 * length]
    
cdef inline void vector_multiply_scalar(DTYPE_t* X, DTYPE_t a, SIZE_t length) nogil:
    "Multiply all elements of the matrix X with a."
    cdef SIZE_t p = 0
    
    for p in range(length):
        X[p] *= a
        
cdef inline void upper_sub_outer_product_eachway(DTYPE_t* X, DTYPE_t* x, DTYPE_t* y, SIZE_t length) nogil:
    "Substract the outer product of x and y as well as y and x from the upper triangular part of X."
    cdef SIZE_t p1 = 0
    cdef SIZE_t p2 = 0
    
    for p1 in range(length):
        for p2 in range(p1, length):
            X[0] -= x[p1] * y[p2] + x[p2] * y[p1]
            X += 1

cdef inline void fill_zeros(DTYPE_t* x, SIZE_t length) nogil:
    "Fill an array with zeros"
    cdef SIZE_t i = 0
    
    for i in range(length):
        x[i] = 0.0

cdef inline SIZE_t upper_n_elements(SIZE_t n) nogil:
    "The number of (diagonal including) elements of an upper triangular nxn matrix."
    return (n * n + n) / 2

cdef inline void vector_add(DTYPE_t* x, DTYPE_t* y, SIZE_t length) nogil:
    "Add vectors y to vector x."
    cdef SIZE_t p = 0
    
    for p in range(length):
        x[p] += y[p]
        
cdef inline void vector_sub(DTYPE_t* x, DTYPE_t* y, SIZE_t length) nogil:
    "Subtract vector y from vector x."
    cdef SIZE_t p = 0
    
    for p in range(length):
        x[p] -= y[p]        

cdef inline void upper_add_sample_autocorrelation_matrix(DTYPE_t* X, DTYPE_t* x, SIZE_t length) nogil:
    "Add the outer product of x with itself to the upper triangular part of X."
    cdef SIZE_t p1 = 0
    cdef SIZE_t p2 = 0
    
    for p1 in range(length):
        for p2 in range(p1, length):
            X[0] += x[p1] * x[p2]
            X += 1
            
cdef inline void upper_sub_sample_autocorrelation_matrix(DTYPE_t* X, DTYPE_t* x, SIZE_t length) nogil:
    "Subtract the outer product of x with itself to the upper triangular part of X."
    cdef SIZE_t p1 = 0
    cdef SIZE_t p2 = 0
    
    for p1 in range(length):
        for p2 in range(p1, length):
            X[0] -= x[p1] * x[p2]
            X += 1
            
#####
# Cpdef functions to expose cdef function for unittesting. Can be deleted if no tests are used.
#####
cpdef int _test_wrapper_n_elements(SIZE_t length):
    return upper_n_elements(length)

cpdef _test_wrapper_upper_add_sample_autocorrelation_matrix(DTYPE_t[::1] X, DTYPE_t[::1] x, SIZE_t length):
    upper_add_sample_autocorrelation_matrix(&X[0], &x[0], length)
    
cpdef _test_wrapper_upper_sub_sample_autocorrelation_matrix(DTYPE_t[::1] X, DTYPE_t[::1] x, SIZE_t length):
    upper_sub_sample_autocorrelation_matrix(&X[0], &x[0], length)    
        
cpdef _test_wrapper_vector_add(DTYPE_t[::1] x, DTYPE_t[::1] y, SIZE_t length):
    vector_add(&x[0], &y[0], length)
    
cpdef _test_wrapper_vector_sub(DTYPE_t[::1] x, DTYPE_t[::1] y, SIZE_t length):
    vector_sub(&x[0], &y[0], length)    
    
cpdef _test_wrapper_vector_multiply_scalar(DTYPE_t[::1] X, DTYPE_t a, SIZE_t length):
    vector_multiply_scalar(&X[0], a, length)

cpdef _test_wrapper_upper_sub_outer_product_eachway(DTYPE_t[::1] X, DTYPE_t[::1] x, DTYPE_t[::1] y, SIZE_t length):
    upper_sub_outer_product_eachway(&X[0], &x[0], &y[0], length)

cpdef _test_wrapper_upper_to_matrix(DTYPE_t[:,::1] X, DTYPE_t[::1] Y, SIZE_t length):
    upper_to_matrix(&X[0][0], &Y[0], length)
