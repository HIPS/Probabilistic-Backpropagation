
# python f-setup.py build_ext --inplace
#   cython f.pyx -> f.cpp
#   g++ -c f.cpp -> f.o
#   g++ -c fc.cpp -> fc.o
#   link f.o fc.o -> f.so

# distutils uses the Makefile distutils.sysconfig.get_makefile_filename()
# for compiling and linking: a sea of options.

# http://docs.python.org/distutils/introduction.html
# http://docs.python.org/distutils/apiref.html  20 pages ...
# http://stackoverflow.com/questions/tagged/distutils+python

import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# from Cython.Build import cythonize

ext_modules = [Extension(
    name="pbp",
    sources=["pbp.pyx", "network.c", "pnorm.c"],
        # extra_objects=["fc.o"],  # if you compile fc.cpp separately
    include_dirs = [numpy.get_include()],  # .../site-packages/numpy/core/include
    language="c",
        libraries= [ "openblas" ],
        extra_compile_args = [ "-O3" ]
        # extra_link_args = "...".split()
    )]

setup(
    name = 'pbp',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
        # ext_modules = cythonize(ext_modules)  ? not in 0.14.1
    # version=
    # description=
    # author=
    # author_email=
    )

# test: import f
