static __device__ bool cubic_root0(real dt, real a, real b, real c, real d, /**/ real *h) {
    const real eps = 1e-8;
    real h1, h2, h3;
    
    if (fabs(a) > eps) { // cubic
        b /= a;
        c /= a;
        d /= a;
            
        int nsol = roots::cubic(b, c, d, &h1, &h2, &h3);

        if (valid_time(dt, h1))             {*h = h1; return true;}
        if (nsol > 1 && valid_time(dt, h2)) {*h = h2; return true;}
        if (nsol > 2 && valid_time(dt, h3)) {*h = h3; return true;}
    }
    else if (fabs(b) > eps) { // quadratic
        if (!roots::quadratic(b, c, d, &h1, &h2)) return false;
        if (valid_time(dt, h1)) {*h = h1; return true;}
        if (valid_time(dt, h2)) {*h = h2; return true;}
    }
    else if (fabs(c) > eps) { // linear
        h1 = -d/c;
        if (valid_time(dt, h1)) {*h = h1; return true;}
    }
    
    return false;
}
