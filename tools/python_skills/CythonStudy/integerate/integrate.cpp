#include <iostream>
#include "integrate.h"

double c_integrate::Integ::integrate(double ub, double lb, double(*func)(double x), int n){
	double dx = (ub - lb) / n;
	double s = 0;
	double i;
			
	for (i = lb; i <= ub; i+=dx){
		s += func(i)*dx;
	}
	return s;
}

c_integrate::Integ::Integ(){};
c_integrate::Integ::~Integ(){};

double c_integrate::fun(double x){
    return x*x;
}

// int main (){
	// double a1 = 0, b1 = 2;
	// int n1 = 20000;
	// c_integrate::Integ integr;
	
	// double areas = integr.integrate(b1,a1,c_integrate::fun,n1);
	// printf("area: %e", areas);
	
	// return 0;
// }