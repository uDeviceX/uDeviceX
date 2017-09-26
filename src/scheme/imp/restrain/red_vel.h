static void report() {
    enum {X, Y, Z};
    int n;
    float v[3];
    restrain::stat(/**/ v, &n);
    MSG("restrain RED: n = %d [% .3e % .3e % .3e]", n, v[X], v[Y], v[Z]);
}

void restrain(const int *cc, int n, long it, /**/ Particle *pp) {
    bool cond;
    int freq;
    restrain::vel(cc, n, RED_COLOR, /**/ pp);

    freq = RESTRAIN_REPORT_FREQ;
    cond = freq > 0 && it % freq == 0;
    if (cond) report();
}
