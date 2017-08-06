namespace h5 {
static int minmax(int lo, int hi, int a) { return min(hi, max(lo, a)); }

static void zero0(float *a, int n) {
    int i;
    for (i = 0; i < n; i ++) a[i] = 0.0;
}

static void zero(float *rho, float *v[3], int n) {
    enum {X, Y, Z};
    zero0(rho, n);
    zero0(v[X], n); zero0(v[Y], n); zero0(v[Z], n);
}

static void avg(Particle *pp, int n, int nc, /**/
                float *rho, float *u[3]) {
    enum {X, Y, Z};
    int c, i, entry;
    float *r, *v;
    zero(rho, u, nc);
    for (i = 0; i < n; ++i) {
        r = pp[i].r;
        v = pp[i].v;
        int index[3] = {
            minmax(0, XS - 1, (int)(floor(r[X])) + XS / 2),
            minmax(0, YS - 1, (int)(floor(r[Y])) + YS / 2),
            minmax(0, ZS - 1, (int)(floor(r[Z])) + ZS / 2)
        };
        entry = index[0] + XS * (index[1] + YS * index[2]);
        rho[entry] += 1;
        for (c = 0; c < 3; ++c) u[c][entry] += v[c];
    }
    for (c = 0; c < 3; ++c)
    for (i = 0; i < nc; ++i)
        u[c][i] = rho[i] ? u[c][i] / rho[i] : 0;
}

static void dump0(Particle *pp, int n, int nc, /*w*/
                  float *rho, float *u[3]) {
#ifndef NO_H5
    enum {X, Y, Z};
    static int id = 0; /* dump id */
    static bool directory_exists = false;
    char path[BUFSIZ];
    const char *names[] = { "density", "u", "v", "w" };

    avg(pp, n, nc, rho, u);
    if (!directory_exists) {
        if (m::rank == 0) os::mkdir(DUMP_BASE "/h5");
        directory_exists = true;
        MC(l::m::Barrier(m::cart));
    }

    sprintf(path, DUMP_BASE "/h5/flowfields-%04d.h5", id++);
    float *data[] = { rho, u[X], u[Y], u[Z] };
    fields(path, data, names, 4);
#endif // NO_H5
}

void dump(Particle *pp, int n) {
#ifndef NO_H5
    enum {X, Y, Z};
    int nc, sz;
    float *rho, *u[3];
    nc = XS * YS * ZS;
    sz = nc*sizeof(rho[0]);
    rho  =  (float*)malloc(sz);
    u[X] = (float*)malloc(sz);
    u[Y] = (float*)malloc(sz);
    u[Z] = (float*)malloc(sz);
    dump0(pp, n, nc, /*w*/ rho, u);
    free(rho); free(u[X]); free(u[Y]); free(u[Z]);
#endif // NO_H5
}
}
