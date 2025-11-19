#include <iostream>
#include <sstream>

#include "C:\Users\victo\Documents\vivi\university\Semester5\ScientificComputing\Anfenger_new\HP_Anfenger\src\simd.hpp"
#include<math.h>
#include <simd_avx.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ASC_HPC;
using std::cout, std::endl;
using ASC_HPC::operator<<;


auto func1 (SIMD<double> a, SIMD<double> b)
{
  return a+b;
}

auto func2 (SIMD<double,4> a, SIMD<double,4> b)
{
  return a+3*b;
}

auto func3 (SIMD<double,4> a, SIMD<double,4> b, SIMD<double,4> c)
{
  return fma(a,b,c);
}


auto load (double * p)
{
  return SIMD<double,2>(p);
}

auto loadMask(double *p, SIMD<mask64, 2> m)
{
  return SIMD<double,2>(p, m);
}

auto testSelect (SIMD<mask64,2> m, SIMD<double,2> a, SIMD<double,2> b)
{
  return select (m, a, b);
}

SIMD<double,2> testHSum (SIMD<double,4> a, SIMD<double,4> b)
{
  return hSum(a,b);
}



int main()
{
  SIMD<double,4> a(1.,2.,3.,4.);
  SIMD<double,4> b(1.0);
  
  cout << "a = " << a << endl;
  cout << "b = " << b << endl;
  cout << "a*b = " << a*b << endl;
  cout << "a+b = " << a+b << endl;

  cout << "HSum(a) = " << hSum(a) << endl;
  cout << "HSum(a,b) = " << hSum(a,b) << endl;
  
  auto sequ = IndexSequence<int64_t, 4>();
  cout << "sequ = " << sequ << endl;
  auto mask = (2 >= sequ);
  cout << "2 >= " << sequ << " = " << mask << endl;

  {
    double a[] = { 10, 10, 10, 10 };
    SIMD<double,4> sa(&a[0], mask);
    cout << "sa = " << sa << endl;
  }

  cout << "select(mask, a, b) = " << select(mask, a,b) << endl;
  cout << M_PI << endl;

//skalar

  {
    double angle = M_PI/4;
    auto [s, c] = sincos(angle);
    cout << "Test sincos(double):" << endl;
    cout << "angle = " << angle << endl;
    cout << "sin(angle) = " << s << endl;
    cout << "cos(angle) = " << c << endl;
  }

//simd 

  {
    SIMD<double,4> angles(0.0, M_PI/2., M_PI, 3.0*M_PI/2.);
    auto [s, c] = sincos(angles);
    cout << "Test sincos(SIMD<double,4>):" << endl;
    cout << "angles = " << angles << endl;
    cout << "sin(angles) = " << s << endl;
    cout << "cos(angles) = " << c << endl;
  }
   

  cout << "\n--- Testing SIMD<double,2> ---" << endl;
  SIMD<double,2> x(5.0, 7.0);
  SIMD<double,2> y(2.0, 3.0);
  cout << "x = " << x << endl;
  cout << "y = " << y << endl;
  cout << "x+y = " << x+y << endl;
  cout << "x*y = " << x*y << endl;
  
  // Test splitting SIMD<double,4> into two SIMD<double,2>
  cout << "\n--- Splitting SIMD<double,4> ---" << endl;
  cout << "a.lo() = " << a.lo() << endl;
  cout << "a.hi() = " << a.hi() << endl;
  
  // Test combining two SIMD<double,2> into SIMD<double,4>
  cout << "\n--- Combining SIMD<double,2> to SIMD<double,4> ---" << endl;
  SIMD<double,4> combined(x, y);
  cout << "SIMD<double,4>(x, y) = " << combined << endl;

  // Test bitonic sort
  cout << "\n--- Bitonic Sort ---" << endl;
  SIMD<double,4> unsorted(4.0, 2.0, 3.0, 1.0);
  cout << "Unsorted: " << unsorted << endl;
  SIMD<double,4> sorted = bitonic_sort(unsorted);
  cout << "Sorted:   " << sorted << endl;
  
  SIMD<double,2> unsorted2(7.0, 5.0);
  cout << "Unsorted SIMD<double,2>: " << unsorted2 << endl;
  SIMD<double,2> sorted2 = bitonic_sort(unsorted2);
  cout << "Sorted SIMD<double,2>:   " << sorted2 << endl;
  
}
