===========================
Configuration of DynStatCov
===========================
In some cases, the source has to be altered to achieve a new configuration.

Changing the data type
----------------------
Locate *dynstatcov.pxd* in the modules source directory, change the line

.. code-block:: cython

    ctypedef np.npy_float64 DTYPE_t # data type
    
according to your wishes and then re-compile as described under :doc:`installation`.

Using DynStatCov from C/C++
---------------------------
By default, the module is not public to C/C++. To change this, Locate *dynstatcov.pyx* as well as *dynstatcov.pxd* in the modules source directory. Change the line

.. code-block:: cython

    cdef class Dynstatcov:
    
to

.. code-block:: cython

    cdef public class Dynstatcov:
   
in both files and then re-compile as described under :doc:`installation`. This will trigger the generation of a header file *.h*, which can be included from C/C++.

