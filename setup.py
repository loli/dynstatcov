#!/usr/bin/env python
# version: 0.1.1

import os
import warnings

# setuptools >= 0.7 supports 'python setup.py develop'
from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, Extension

#### CONSTANTS
WARN_CYTHON_NOT_FOUND = 'Could not locate Cython. Falling back to already Cython-complied .c file instead.'

#### PRE-CODE (to figure out, whether Cython is installed or not)
try:
    from Cython.Distutils import build_ext
    CYTHON_FOUND = True
except ImportError as e:
    from distutils.command.build_ext import build_ext
    warnings.warn(WARN_CYTHON_NOT_FOUND)
    CYTHON_FOUND = False

#### FUNCTIONS
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

### MAIN SETUP
dynstatcov = Extension('dynstatcov',
                       sources = ['dynstatcov/dynstatcov.pyx', 'dynstatcov/dynstatcov.pxd'],
                       #include_dirs=[numpy.get_include()],
                       extra_compile_args=["-O3"])

configuration = {
    'name' : 'DynStatCov',
    'version' : '0.1.2', # major.minor.micro
    'author' : 'Oskar Maier',
    'author_email' : 'oskar.maier@googlemail.com',
    'url' : 'https://github.com/loli/dynstatcov',
    'license' : 'LICENSE.txt',
    'keywords' : 'dynamic statistical co-variance matrix update cython',
    'long_description' : read('README.rst'),
    
    'classifiers' : [
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: Other Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        #'Operating System :: MacOS :: MacOS X',
        #'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',        
        'Programming Language :: Cython',
        'Programming Language :: C',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development'
    ],
    
    'install_requires' : [
        "numpy >= 1.6.1",
    ],
                 
    'zip_safe' : False,
    
    # extention building part
    'ext_modules' : [dynstatcov],
    'cmdclass' : {'build_ext': build_ext}
}

if not CYTHON_FOUND:
    dynstatcov.sources = ['dynstatcov/dynstatcov.c']

setup(**configuration)

