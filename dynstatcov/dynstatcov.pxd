# python imports

# cython imports
cimport numpy as np

# extern cdefs

# type definitions
ctypedef np.npy_float64 DTYPE_t # data type
ctypedef np.npy_intp SIZE_t     # type for indices and counters

cdef class Dynstatcov:
    # Dynamically updateable statistical co-variance matrix.
    
    # internal structure
    cdef DTYPE_t* cov           # upper triangular part of the co-variance matrix
    cdef DTYPE_t* squaresum     # upper triangular part of the sum of all samples outer product
    cdef DTYPE_t* sum           # sum of all samples
    cdef SIZE_t n_samples       # number of samples from which the co-variance matrix is computed
    cdef SIZE_t n_features      # number of elements per samples
    cdef SIZE_t n_upper         # elements in the upper triangular matrix
    
    cdef DTYPE_t* __mean        # private member

    # methods
    cpdef update(self, DTYPE_t[::1] x, int subtract = *) # update the co-variance matrix with a new sample or remove one
    cpdef np.ndarray get_cov(self)                       # return the (full) co-variance matrix
    cpdef np.ndarray get_cov_tri(self)                   # return the upper triangular part of the co-variance matrix
    cpdef int get_n_samples(self)                        # return the number of samples used for compute the co-variance matrix
    
    cdef void __update_add(self, DTYPE_t* x) nogil       # internal, no-gil update method (addition)
    cdef void __update_sub(self, DTYPE_t* x) nogil       # internal, no-gil update method (subtraction)
    cdef void __compute_covariance_matrix(self) nogil    # trigger the computation of the co-variance matrix
