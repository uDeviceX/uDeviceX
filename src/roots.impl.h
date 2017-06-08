namespace roots
{
#define _DH_ __device__ __host__
#define SWAP(a,b) do { auto tmp = b ; b = a ; a = tmp ; } while(0)

template<typename real>
_DH_ bool quadratic(real a, real b, real c, real *h0, real *h1)
{
    const int sgnb = b > 0 ? 1 : -1;
    const real D = b*b - 4*a*c;

    if (D < 0) return false;
        
    *h0 = (-b - sgnb * sqrt(D)) / (2 * a);
    *h1 = c / (a * *h0);
        
    if (*h0 > *h1) SWAP(*h0, *h1);
    return true;
}

/* code copied from gsl library; gsl_poly_solve_cubic (see end of file for initial header)
   https://www.gnu.org/software/gsl/
   slightly adapted 
*/

template <typename real>
_DH_ int cubic(real a, real b, real c, 
               real *x0, real *x1, real *x2)
{
    real q = (a * a - 3 * b);
    real r = (2 * a * a * a - 9 * a * b + 27 * c);

    real Q = q / 9;
    real R = r / 54;

    real Q3 = Q * Q * Q;
    real R2 = R * R;

    real CR2 = 729 * r * r;
    real CQ3 = 2916 * q * q * q;

    if (R == 0 && Q == 0)
    {
        *x0 = - a / 3 ;
        *x1 = - a / 3 ;
        *x2 = - a / 3 ;
        return 3 ;
    }
    else if (CR2 == CQ3) 
    {
        /* this test is actually R2 == Q3, written in a form suitable
           for exact computation with integers */

        /* Due to finite precision some real roots may be missed, and
           considered to be a pair of complex roots z = x +/- epsilon i
           close to the real axis. */

        real sqrtQ = sqrt (Q);

        if (R > 0)
        {
            *x0 = -2 * sqrtQ  - a / 3;
            *x1 = sqrtQ - a / 3;
            *x2 = sqrtQ - a / 3;
        }
        else
        {
            *x0 = - sqrtQ  - a / 3;
            *x1 = - sqrtQ - a / 3;
            *x2 = 2 * sqrtQ - a / 3;
        }
        return 3 ;
    }
    else if (R2 < Q3)
    {
        real sgnR = (R >= 0 ? 1 : -1);
        real ratio = sgnR * sqrt (R2 / Q3);
        real theta = acos (ratio);
        real norm = -2 * sqrt (Q);
        *x0 = norm * cos (theta / 3) - a / 3;
        *x1 = norm * cos ((theta + 2.0 * M_PI) / 3) - a / 3;
        *x2 = norm * cos ((theta - 2.0 * M_PI) / 3) - a / 3;
      
        /* Sort *x0, *x1, *x2 into increasing order */

        if (*x0 > *x1)
        SWAP(*x0, *x1) ;
      
        if (*x1 > *x2)
        {
            SWAP(*x1, *x2) ;
          
            if (*x0 > *x1)
            SWAP(*x0, *x1) ;
        }
      
        return 3;
    }
    else
    {
        real sgnR = (R >= 0 ? 1 : -1);
        real A = -sgnR * pow (fabs (R) + sqrt (R2 - Q3), 1.0/3.0);
        real B = Q / A ;
        *x0 = A + B - a / 3;
        return 1;
    }
}

/* poly/solve_cubic.c
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2007, 2009 Brian Gough
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* solve_cubic.c - finds the real roots of x^3 + a x^2 + b x + c = 0 */
}
