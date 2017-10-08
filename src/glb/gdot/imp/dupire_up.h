static float ff(long it, long nsteps) {
    float gd = 0;
    int i, n = 8;
    for (i = 1; i <= n; i++)
        if (n*it < i*nsteps) {
            gd =  2.0*i/n*gamma_dot;
            break
        }
    return gd;
}

float f0() {               return ff(0, 1); }
float  f(long s, long e) { return ff(s, e); }
