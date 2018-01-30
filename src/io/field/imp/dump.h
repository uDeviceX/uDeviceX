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

static void avg(const Coords *coords, Particle *pp, int n, int nc, /**/
                float *rho, float *u[3]) {
    enum {X, Y, Z};
    int c, i, entry;
    float *r, *v;
    int Lx, Ly, Lz;

    Lx = xs(coords);
    Ly = ys(coords);
    Lz = zs(coords);

    zero(rho, u, nc);
    for (i = 0; i < n; ++i) {
        r = pp[i].r;
        v = pp[i].v;
        int index[3] = {
            minmax(0, Lx - 1, (int)(floor(r[X])) + Lx / 2),
            minmax(0, Ly - 1, (int)(floor(r[Y])) + Ly / 2),
            minmax(0, Lz - 1, (int)(floor(r[Z])) + Lz / 2)
        };
        entry = index[0] + Lx * (index[1] + Ly * index[2]);
        rho[entry] += 1;
        for (c = 0; c < 3; ++c) u[c][entry] += v[c];
    }
    for (c = 0; c < 3; ++c)
    for (i = 0; i < nc; ++i)
        u[c][i] = rho[i] ? u[c][i] / rho[i] : 0;
}

static void dump0(const Coords *coords, MPI_Comm cart, Particle *pp, int n, int nc, /*w*/
                  float *rho, float *u[3]) {
    enum {X, Y, Z};
    static int id = 0; /* dump id */
    static bool directory_exists = false;
    char path[BUFSIZ];
    const char *names[] = { "density", "u", "v", "w" };

    avg(coords, pp, n, nc, rho, u);
    if (!directory_exists) {
        if (m::is_master(cart))
            UC(os_mkdir(DUMP_BASE "/h5"));
        directory_exists = true;
        MC(m::Barrier(cart));
    }

    sprintf(path, DUMP_BASE "/h5/flowfields-%04d.h5", id++);
    float *data[] = { rho, u[X], u[Y], u[Z] };
    UC(h5_write(coords, cart, path, data, names, 4));
    if (m::is_master(cart))
        xmf_write(path, names, 4, xs(coords), ys(coords), zs(coords));
}

void dump(const Coords *coords, MPI_Comm cart, Particle *pp, int n) {
    enum {X, Y, Z};
    int nc, sz;
    float *rho, *u[3];
    nc = xs(coords) * ys(coords) * zs(coords);
    sz = nc*sizeof(rho[0]);

    UC(emalloc(sz, (void**) &rho));
    UC(emalloc(sz, (void**) &u[X]));
    UC(emalloc(sz, (void**) &u[Y]));
    UC(emalloc(sz, (void**) &u[Z]));
    dump0(coords, cart, pp, n, nc, /*w*/ rho, u);
    efree(rho); efree(u[X]); efree(u[Y]); efree(u[Z]);
}
