namespace h5 {
static int minmax(int lo, int hi, int a) { return min(hi, max(lo, a)); }

static void dump0(Particle *pp, int n,
                  int ncells, /*w*/
                  std::vector<float> rho,
                  std::vector<float> u[3]) {
#ifndef NO_H5
    enum {X, Y, Z};
    static int id = 0; /* dump id */
    static bool directory_exists = false;

    char path[BUFSIZ];
    const char *names[] = { "density", "u", "v", "w" };
    int c, i, entry;
    float *r, *v;
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
    for (i = 0; i < ncells; ++i)
        u[c][i] = rho[i] ? u[c][i] / rho[i] : 0;

    if (!directory_exists) {
        if (m::rank == 0) os::mkdir(DUMP_BASE "/h5");
        directory_exists = true;
        MC(l::m::Barrier(m::cart));
    }

    sprintf(path, DUMP_BASE "/h5/flowfields-%04d.h5", id++);
    float *data[] = { rho.data(), u[X].data(), u[Y].data(), u[Z].data() };
    fields(path, data, names, 4);
#endif // NO_H5
}

void dump(Particle *pp, int n) {
#ifndef NO_H5
    enum {X, Y, Z};
    int nc;
    nc = XS * YS * ZS;
    std::vector<float> rho(nc), u[3];
    u[X].resize(nc);
    u[Y].resize(nc);
    u[Z].resize(nc);
    dump0(pp, n, nc, /*w*/ rho, u);
#endif // NO_H5
}
}
