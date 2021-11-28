cdef extern from "integrate.h" namespace "c_integrate":
    cdef cppclass Integ:
	
        Integ() except +
        double integrate(double ub, double lb, double(*func)(double x), int n)
		
    double fun(double x)
	
	
	