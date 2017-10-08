static void report0() {
    enum {X, Y, Z};
    int n;
    float v[3];
    restrain::stat(/**/ &n, v);
    MSG("restrain RBC: n = %d [% .3e % .3e % .3e]", n, v[X], v[Y], v[Z]);
}

static void report(int it) {
    bool cond;
    int freq;
    freq = RESTRAIN_REPORT_FREQ;
    cond = freq > 0 && it % freq == 0;
    if (cond) report0();
}

void restrain(const int *cc, NN nn, long it, /**/ QQ qq) {
    restrain::grey::vel(cc, RED_COLOR, nn.r, /**/ qq.r);
    report(it);
}
