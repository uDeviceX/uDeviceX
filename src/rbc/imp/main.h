static void forces(int nc, const Texo<float2> texvert, const Texo<int4> textri, const Texo<int> texadj0, const Texo<int> texadj1, Force *ff, float* av) {
    if (nc <= 0) return;

    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);
    int md, nt, nv;
    md = RBCmd;
    nt = RBCnt;
    nv = RBCnv;

    Dzero(av, 2*nc);
    KL(dev::area_volume, (avBlocks, avThreads), (nt, nv, texvert, textri, av));
    KL(dev::force, (k_cnf(nc*nv*md)), (md, nv, texvert, texadj0, texadj1, nc, av, (float*)ff));
}

void forces(const Quants q, const TicketT t, /**/ Force *ff) {
    forces(q.nc, t.texvert, t.textri, t.texadj0, t.texadj1, /**/ ff, q.av);
}

