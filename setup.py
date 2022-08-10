from distutils.core import setup
from xml.etree.ElementInclude import include
from Cython.Build import cythonize
import numpy
#python setup.py build_ext --inplace
include_dirs = [numpy.get_include()]
setup(name='cutils', include_dirs = include_dirs,  ext_modules=cythonize("cutils.pyx"))