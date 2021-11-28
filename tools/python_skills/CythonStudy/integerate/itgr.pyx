#-*-coding: utf-8-*-
#distutils: language = c++

from integrate cimport Integ
from integrate cimport fun

	
cdef class PyInteg():
    cdef Integ c_integ
	
    def __cinit__(self):
	
        self.c_integ = Integ()
		
    def pyintegrate(self, double ub, double lb, int n):
        return self.c_integ.integrate(ub, lb, fun, n)
		
    def f(self, double x):
        return x*x
		
