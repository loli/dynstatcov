import functools

import nose
import numpy
from numpy.testing import assert_allclose, assert_array_equal, assert_equal, assert_raises

from dynstatcov import dynstatcov

####
# Constants
####
DTYPE_t = numpy.float64
SIZE_t = numpy.intp

####
# Tests for functionality exposed to Python
####
class TestDynstatcov(object):

    def setup(self): # called before each test method
        self.X = numpy.array([[3, 4, 1, 2],
                              [4, 4, 0, 3],
                              [1, 3, 4, 2]], dtype=DTYPE_t)
        self.y = numpy.array([3, 4, 1, 2], dtype=DTYPE_t)
        self.dsc = dynstatcov.Dynstatcov(self.X)

    def test_setup_wrong_dimensionality_1d(self):
        X = numpy.asarray([1,2,3] , dtype=DTYPE_t)
        assert_raises(ValueError, dynstatcov.Dynstatcov, X)
        
    def test_setup_wrong_dimensionality_3d(self):
        X = numpy.asarray([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=DTYPE_t)
        assert_raises(ValueError, dynstatcov.Dynstatcov, X)        

    def test_setup_empty(self):
        X = numpy.asarray([[],[]] , dtype=DTYPE_t)
        assert_raises(ValueError, dynstatcov.Dynstatcov, X)
        
    def test_setup_single_observation(self):
        # meant to signal: Exception ZeroDivisionError: 'float division' in 'dynstatcov.dynstatcov.Dynstatcov.__compute_covariance_matrix' ignored
        expected_result = numpy.asarray([[ 0.,  0.,  0.],
                                         [ 0.,  0.,  0.],
                                         [ 0.,  0.,  0.]], dtype=DTYPE_t)
        X = numpy.asarray([[1, 2, 3]] , dtype=DTYPE_t)
        dsc = dynstatcov.Dynstatcov(X)
        result = dsc.get_cov()
        assert_allclose(result, expected_result)

    def test_get_n_samples(self):
        expected_result = self.X.shape[0]
        result = self.dsc.get_n_samples()
        assert result == expected_result

    def test_get_cov(self):
        expected_result = numpy.array([[ 2.33333333,  0.83333333, -3.16666667,  0.66666667],
                                       [ 0.83333333,  0.33333333, -1.16666667,  0.16666667],
                                       [-3.16666667, -1.16666667,  4.33333333, -0.83333333],
                                       [ 0.66666667,  0.16666667, -0.83333333,  0.33333333]], dtype=DTYPE_t)
        result = self.dsc.get_cov()
        assert_allclose(result, expected_result)
        
    def test_get_cov_tri(self):
        expected_result = numpy.array([ 2.33333333,  0.83333333, -3.16666667,  0.66666667,\
                                       0.33333333, -1.16666667,  0.16666667,  4.33333333, \
                                       -0.83333333,  0.33333333], dtype=DTYPE_t)
        result = self.dsc.get_cov_tri()
        assert_allclose(result, expected_result)
        
    def test_udpate_add_get_cov(self):
        expected_result = numpy.array([[ 1.58333333,  0.58333333, -2.16666667,  0.41666667],
                                       [ 0.58333333,  0.25      , -0.83333333,  0.08333333],
                                       [-2.16666667, -0.83333333,  3.        , -0.5       ],
                                       [ 0.41666667,  0.08333333, -0.5       ,  0.25      ]], dtype=DTYPE_t)
        self.dsc.update(self.y)
        result = self.dsc.get_cov()
        assert_allclose(result, expected_result)
        
    def test_udpate_add_sub_get_cov(self):
        expected_result = numpy.array([[ 2.33333333,  0.83333333, -3.16666667,  0.66666667],
                                       [ 0.83333333,  0.33333333, -1.16666667,  0.16666667],
                                       [-3.16666667, -1.16666667,  4.33333333, -0.83333333],
                                       [ 0.66666667,  0.16666667, -0.83333333,  0.33333333]], dtype=DTYPE_t)
        self.dsc.update(self.y)
        self.dsc.update(self.y, 1)
        result = self.dsc.get_cov()
        assert_allclose(result, expected_result)           
        
    def test_udpate_get_n_samples_add(self):
        expected_result = self.X.shape[0] + 1
        self.dsc.update(self.y)
        result = self.dsc.get_n_samples()
        assert result == expected_result
        
    def test_udpate_get_n_samples_sub(self):
        expected_result = self.X.shape[0] - 1
        self.dsc.update(self.y, 1)
        result = self.dsc.get_n_samples()
        assert result == expected_result
        
    def test_cov_ownership(self):
        cov = self.dsc.get_cov()
        assert cov.flags.owndata       
        
    def test_cov_tri_ownership(self):
        cov_tri = self.dsc.get_cov_tri()
        assert cov_tri.flags.owndata  
        
    def test_numpy_base(self):
        "Test against numpy cov-matrix."
        expected_result = TestDynstatcov.__numpycov(self.X)
        result = self.dsc.get_cov()
        assert_allclose(result, expected_result)
        
    def test_numpy_update_add(self):
        expected_result = TestDynstatcov.__numpycov(numpy.vstack((self.X, self.y)))
        self.dsc.update(self.y)
        result = self.dsc.get_cov()
        assert_allclose(result, expected_result)
        
    def test_numpy_update_sub(self):
        expected_result = TestDynstatcov.__numpycov(self.X[:-1])
        self.dsc.update(self.X[-1], 1)
        result = self.dsc.get_cov()
        assert_allclose(result, expected_result)        
        
    def test_sample_base(self):
        "Test against python computed cov-matrix."
        expected_result = TestDynstatcov.__samplecov(self.X)
        result = self.dsc.get_cov()
        assert_allclose(result, expected_result)
        
    def test_sample_update_add(self):
        expected_result = TestDynstatcov.__samplecov(numpy.vstack((self.X, self.y)))
        self.dsc.update(self.y)
        result = self.dsc.get_cov()
        assert_allclose(result, expected_result)
        
    def test_sample_update_sub(self):
        expected_result = TestDynstatcov.__samplecov(self.X[:-1])
        self.dsc.update(self.X[-1], 1)
        result = self.dsc.get_cov()
        assert_allclose(result, expected_result)          
        
    @staticmethod
    def __numpycov(X):
        q = numpy.mean(X, axis=0)
        Xc = X - q
        return numpy.cov(Xc, rowvar=0)
        
    @staticmethod
    def __samplecov(X):
        n = X.shape[0]
        q = numpy.mean(X, axis=0)
        Xc = X - q
        return 1./(n-1) * Xc.T.dot(Xc)        

####
# Conditional tests for cdef functions.
# If test wrappers not available from module dynstatcov, we assume they have been removed to reduce file size and skip the tests.
####
# decorator to convert caught AttributeErrors to SkipTest exceptions
def skip_on_attribute_error(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        try:
            fun(*args, **kwargs)
        except AttributeError:
            raise nose.SkipTest
    return wrapper

@skip_on_attribute_error
def test_upper_to_matrix():
    length = 10
    X = numpy.zeros((length, length), dtype=DTYPE_t)
    Y = numpy.asarray(range((length * length + length)/2), dtype=DTYPE_t)
    
    # compute expected results (first filling upper triangle including the diagonal with values, then mirror the same minus the diagonal down)
    expected_result = numpy.zeros((length, length), dtype=DTYPE_t)
    expected_result[numpy.triu(numpy.ones((length, length))).astype(numpy.bool)] = Y
    expected_result.T[numpy.triu(numpy.ones((length, length)), k=1).astype(numpy.bool)] = \
        expected_result[numpy.triu(numpy.ones((length, length)), k=1).astype(numpy.bool)]
        
    # run and test
    dynstatcov._test_wrapper_upper_to_matrix(X, Y, length)
    assert_array_equal(X, expected_result)

@skip_on_attribute_error
def test_vector_multiply_scalar():
    length = 10
    x = numpy.random.random(length).astype(DTYPE_t)
    a = numpy.random.random()
    
    expected_result = x * a
    dynstatcov._test_wrapper_vector_multiply_scalar(x, a, length)
    assert_array_equal(x, expected_result)

@skip_on_attribute_error
def test_upper_sub_outer_product_eachway():
    length = 3
    x = numpy.asarray(range(length), dtype=DTYPE_t) + 1
    y = x[::-1].copy()
    X = numpy.zeros((length * length + length)/2, dtype=DTYPE_t)
    expected_result = numpy.zeros(X.shape[0], dtype=DTYPE_t)
    
    # compute expected results
    c = 0
    for i in range(1, length + 1):
        for j in range(i, length + 1):
            expected_result[c] = -1 * (i * (length - j + 1) + j * (length - i + 1))
            c += 1      
    
    # run and test
    dynstatcov._test_wrapper_upper_sub_outer_product_eachway(X, x, y, length)
    assert_array_equal(X, expected_result)

@skip_on_attribute_error
def test_vector_add():
    length = 10
    x = numpy.asarray(range(length), dtype=DTYPE_t)
    y = x[::-1].copy()
    expected_result = numpy.asarray([length-1] * length, dtype=DTYPE_t)
    
    dynstatcov._test_wrapper_vector_add(x, y, length)
    assert_array_equal(x, expected_result)
    
@skip_on_attribute_error
def test_vector_sub():
    length = 10
    x = numpy.asarray(range(length), dtype=DTYPE_t)
    y = x[::-1].copy()
    expected_result = numpy.asarray(range(-1 * length + 1, length, 2), dtype=DTYPE_t)
    
    dynstatcov._test_wrapper_vector_sub(x, y, length)
    assert_array_equal(x, expected_result)    
    
@skip_on_attribute_error
def test_upper_add_sample_autocorrelation_matrix():
    length = 5
    x = numpy.asarray(range(length), dtype=DTYPE_t) + 1
    X = numpy.zeros((length * length + length)/2, dtype=DTYPE_t)
    expected_result = numpy.zeros(X.shape[0], dtype=DTYPE_t)
    
    # compute expected results
    c = 0
    for i in range(1, length + 1):
        for j in range(0, length - i + 1):
            expected_result[c] = (j + i) * i
            c += 1    
    
    # run and test
    dynstatcov._test_wrapper_upper_add_sample_autocorrelation_matrix(X, x, length)
    assert_array_equal(X, expected_result)    
    
@skip_on_attribute_error
def test_upper_sub_sample_autocorrelation_matrix():
    length = 5
    x = numpy.asarray(range(length), dtype=DTYPE_t) + 1
    X = numpy.zeros((length * length + length)/2, dtype=DTYPE_t)
    expected_result = numpy.zeros(X.shape[0], dtype=DTYPE_t)
    
    # compute expected results
    c = 0
    for i in range(1, length + 1):
        for j in range(0, length - i + 1):
            expected_result[c] = -1 * (j + i) * i
            c += 1    
    
    # run and test
    dynstatcov._test_wrapper_upper_sub_sample_autocorrelation_matrix(X, x, length)
    assert_array_equal(X, expected_result)
    
@skip_on_attribute_error
def test_upper_n_elements():
    length = 99
    expected_result = 0
        
    for i in range(1, length + 1):
        expected_result += i
        
    result = dynstatcov._test_wrapper_n_elements(length)
    assert_equal(result, expected_result)
    