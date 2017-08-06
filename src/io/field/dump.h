namespace h5 {
void dump(Particle *pp, int n) {
#ifndef NO_H5
    static int id = 0; /* dump id */
    static bool directory_exists = false;

    char path[BUFSIZ];
    const char *names[] = { "density", "u", "v", "w" };
    int ncells;
    int c, i, entry;
    float *r, *v;
    ncells = XS * YS * ZS;
    std::vector<float> rho(ncells), u[3];

    for (c = 0; c < 3; ++c) u[c].resize(ncells);

    for (i = 0; i < n; ++i) {
        r = pp[i].r;
        v = pp[i].v;
        int index[3] = {
             max(0, min(XS - 1, (int)(floor(r[0])) + XS / 2)),
             max(0, min(YS - 1, (int)(floor(r[1])) + YS / 2)),
             max(0, min(ZS - 1, (int)(floor(r[2])) + ZS / 2))
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
    float *data[] = { rho.data(), u[0].data(), u[1].data(), u[2].data() };
    fields(path, data, names, 4);
#endif // NO_H5
}
}
