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
