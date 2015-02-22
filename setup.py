#!/usr/bin/env python
# version: 0.1.0

import os
from distutils.core import setup
from Cython.Build import cythonize

#### FUNCTIONS
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

### MAIN SETUP
setup(
    name = 'DynStatCov',
    version = '0.1.0', # major.minor.micro
    author = 'Oskar Maier',
    author_email = 'oskar.maier@googlemail.com',
    url='https://github.com/loli/dynstatcov',
    license='LICENSE.txt',
    keywords='dynamic statistical co-variance matrix update cython',
    long_description=read('README.rst'),
    
    classifiers=[
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
    
    #install_requires=[
    #    "numpy >= 1.6.1",
    #],
    
    packages = [
        'dynstatcov'
    ],
    
    ext_modules = cythonize("dynstatcov/dynstatcov.pyx")
)

