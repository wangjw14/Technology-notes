#ifndef INTEGRATE_H
#define INTEGRATE_H

namespace c_integrate{
	class Integ{
	public:
		Integ();
		~Integ();
        double integrate(double ub, double lb, double(*func)(double x), int n);
		      		
};
double fun(double x); 
}

#endif //INTEGRATE_H
