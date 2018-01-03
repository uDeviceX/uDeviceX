static int minmax(int lo, int hi, int x) {
    return \
        x <= lo ? lo :
        x >= hi ? hi :
        x;
}

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

static void dump0(MPI_Comm cart, Particle *pp, int n, int nc, /*w*/
                  float *rho, float *u[3]) {
    enum {X, Y, Z};
    static int id = 0; /* dump id */
    static bool directory_exists = false;
    char path[BUFSIZ];
    const char *names[] = { "density", "u", "v", "w" };

    avg(pp, n, nc, rho, u);
    if (!directory_exists) {
        if (m::is_master(cart))
            UC(os::mkdir(DUMP_BASE "/h5"));
        directory_exists = true;
        MC(m::Barrier(cart));
    }

    sprintf(path, DUMP_BASE "/h5/flowfields-%04d.h5", id++);
    float *data[] = { rho, u[X], u[Y], u[Z] };
    UC(h5::write(cart, path, data, names, 4, XS, YS, ZS));
    if (m::is_master(cart))
        xmf::write(path, names, 4, XS, YS, ZS);
}

void dump(MPI_Comm cart, Particle *pp, int n) {
    enum {X, Y, Z};
    int nc, sz;
    float *rho, *u[3];
    nc = XS * YS * ZS;
    sz = nc*sizeof(rho[0]);

    UC(emalloc(sz, (void**) &rho));
    UC(emalloc(sz, (void**) &u[X]));
    UC(emalloc(sz, (void**) &u[Y]));
    UC(emalloc(sz, (void**) &u[Z]));
    dump0(cart, pp, n, nc, /*w*/ rho, u);
    efree(rho); efree(u[X]); efree(u[Y]); efree(u[Z]);
}
