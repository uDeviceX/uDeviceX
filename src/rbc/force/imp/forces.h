static void forces0(int nc, const Texo<float2> texvert, const Texo<int> texadj0, const Texo<int> texadj1, float* av, Force *ff) {
    int md, nv;
    md = RBCmd;
    nv = RBCnv;
    KL(dev::force, (k_cnf(nc*nv*md)), (md, nv, texvert, texadj0, texadj1, nc, av, (float*)ff));
}

void forces(const Quants q, const TicketT t, /**/ Force *ff) {
    if (q.nc <= 0) return;
    area_volume(q.nc, t.texvert, t.textri, /**/ q.av);
    forces0(q.nc, t.texvert, t.texadj0, t.texadj1, q.av, /**/ ff);
}
