static void report() {
    enum {X, Y, Z};
    float v[3];
    restrain::vcm(/**/ v);
    MSG("restrain RED velocity: [% .2e % .2e % .2e]", v[X], v[Y], v[Z]);
}

void restrain(const int *cc, int n, long it, /**/ Particle *pp) {
    restrain::vel(cc, n, RED_COLOR, /**/ pp);
    if (it % RESTRAIN_PAR_REPORT_FREQ == 0) report();
}
