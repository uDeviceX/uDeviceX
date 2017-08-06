void H5FieldDump::dump(Particle *p, int n) {
#ifndef NO_H5
    static int id = 0; /* dump id */
     
    const int ncells = XS * YS * ZS;
    std::vector<float> rho(ncells), u[3];
    for(int c = 0; c < 3; ++c)
    u[c].resize(ncells);
    for(int i = 0; i < n; ++i) {
        const int cellindex[3] = {
            max(0, min(XS - 1, (int)(floor(p[i].r[0])) + XS / 2)),
            max(0, min(YS - 1, (int)(floor(p[i].r[1])) + YS / 2)),
            max(0, min(ZS - 1, (int)(floor(p[i].r[2])) + ZS / 2))
        };
        const int entry = cellindex[0] + XS * (cellindex[1] + YS * cellindex[2]);
        rho[entry] += 1;
        for(int c = 0; c < 3; ++c) u[c][entry] += p[i].v[c];
    }

    for(int c = 0; c < 3; ++c)
    for(int i = 0; i < ncells; ++i)
    u[c][i] = rho[i] ? u[c][i] / rho[i] : 0;

    const char * names[] = { "density", "u", "v", "w" };
    if (!directory_exists) {
        if (m::rank == 0) os::mkdir(DUMP_BASE "/h5");
        directory_exists = true;
        MC(l::m::Barrier(m::cart));
    }

    char filepath[512];
    sprintf(filepath, DUMP_BASE "/h5/flowfields-%04d.h5", id++);
    float * data[] = { rho.data(), u[0].data(), u[1].data(), u[2].data() };
    fields(filepath, data, names, 4);
#endif // NO_H5
}
