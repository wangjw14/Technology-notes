#-*-coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [Extension("itgr", sources=["itgr.pyx","integrate.cpp"],
    language="c++")]

setup(
    name="itgr",
    ext_modules = cythonize(ext_modules)
)


# setup(
    # ext_modules=cythonize(
	# [Extension("integrate",
	# sources=["integrate.pyx", 'integrate.cpp'],
	# language="c++")])
# )


