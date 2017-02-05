#include "stdio.h"
#include <iostream>
#include <vector>
#include <CGAL/Simple_cartesian.h>

typedef CGAL::Simple_cartesian<float> Kernel;
typedef Kernel::Point_2 Point_2;
typedef std::vector<float> TVec;

void hello_a(TVec& sol_xx, TVec& sol_yy, TVec& sol_zz,
	     TVec& rbc_xx, TVec& rbc_yy, TVec& rbc_zz) {
  printf("(hello_a.cpp) hello_a is called\n");

  auto nsol = sol_xx.size();
  Point_2 p(1, 42);
  printf("(hello_a.cpp) %g %g\n", p[0], p[1]);
  
}
