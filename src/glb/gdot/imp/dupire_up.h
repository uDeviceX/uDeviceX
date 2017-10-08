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

static void report0(float gd) {
    MSG("GDOT_DUPIRE_UP: gd = %g\n", gd);
}

static void report(float gd) {
    bool cond;
    int freq;
    freq = GDOT_REPORT_FREQ;
    cond = freq > 0 && it % freq == 0;
    if (cond) report0(gd);
}

float f0() {               return ff(0, 1); }
float  f(long s, long e) {
    float gd;
    gd = ff(s, e);
    report(gd);
    return gd;
}
