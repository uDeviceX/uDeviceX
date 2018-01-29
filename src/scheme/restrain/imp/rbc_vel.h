static void report0() {
    enum {X, Y, Z};
    int n;
    float v[3];
    stats(/**/ &n, v);
    msg_print("restrain RBC: n = %d [% .3e % .3e % .3e]", n, v[X], v[Y], v[Z]);
}

static void report(int it) {
    bool cond;
    int freq;
    freq = RESTRAIN_REPORT_FREQ;
    cond = freq > 0 && it % freq == 0;
    if (cond) report0();
}

void scheme_restrain_apply(MPI_Comm comm, const int*, long it, /**/ SchemeQQ qq) {
    grey::vel(comm, qq.rn, /**/ qq.r);
    report(it);
}
