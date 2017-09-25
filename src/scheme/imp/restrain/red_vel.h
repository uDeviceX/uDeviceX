static void report() {
    enum {X, Y, Z};
    float v[3];
    restrain::vcm(/**/ v);
    MSG("vcm: [% .2e % .2e % .2e]", v[X], v[Y], v[Z]);
}

void restrain(const int *cc, int n, /**/ Particle *pp) {
    restrain::vel(cc, n, RED_COLOR, /**/ pp);
    report();
}
