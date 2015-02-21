===============================
Installing/Compiling DynStatCov
===============================

Getting the source code
-----------------------
https://github.com/loli/dynstatcov/

From source without Cython
--------------------------
The source comes with a ready-to-compile `.c` file. Simply run

.. code-block:: bash

    python setup.py build
    python setup.py install [requires root]
    
respectively

.. code-block:: bash

    python setup.py install --user
    
to install with user priviledges only. Not Cython required.

From source with Cython
-----------------------
After changes to the source files, a Cython re-compliation becomes necessary. This is automatically taken care of when building the module as described above.

Alternatively, the module can be build in-place for testing:

.. code-block:: bash

    python setup.py build_ext --inplace
    
which will generate a module file called `dynstatcov.so` in the directory with the Cython source files.
