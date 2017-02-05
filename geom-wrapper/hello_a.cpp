#include "stdio.h"
#include <iostream>
#include <CGAL/Simple_cartesian.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_2 Point_2;

void hello_a() {
  printf("(hello_a.cpp) hello_a is called\n");

  Point_2 p(1, 42);
  printf("(hello_a.cpp) %g %g\n", p[0], p[1]);
  
}
