static float ff(long it, long nsteps) {
    float gd = 0;
    int i, n = 8;
    for (i = 1; i <= n; i++)
        if (n*it < i*nsteps) {
            gd =  2.0*i/n*gamma_dot;
            break;
        }
    return gd;
}

static void report0(long i, long e, float gd) {
    MSG("GDOT_DUPIRE_UP: gd = %g : step %d/%d", gd, i, e);
}

static void report(long i, long e, float gd) {
    bool cond;
    int freq;
    freq = GDOT_REPORT_FREQ;
    cond = freq > 0 && i % freq == 0;
    if (cond) report0(i, e, gd);
}

float f0() {               return ff(0, 1); }
float  f(long i, long e) {
    float gd;
    gd = ff(i, e);
    report(i, e, gd);
    return gd;
}
