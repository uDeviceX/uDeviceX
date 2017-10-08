static float ff(long it, long nsteps) {
    float gd = 0;
    int i, n = 8;
    for (i = 1; i <= n; i++)
        if (n*it < i*nsteps) {
            gd =  2.0*(n-i+1)/n*gamma_dot;
            break;
        }
    return gd;
}

static void report0(long i, long e, float gd) {
    MSG("GDOT_DUPIRE_DOWN: gd = %6.2g : step %07ld/%07ld", gd, i, e);
}
