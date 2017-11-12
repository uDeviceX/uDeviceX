static void apply0(int nc, const Texo<float2> vert, const Texo<int> adj0, const Texo<int> adj1, float* av, /**/ Force *ff){
    int md, nv;
    md = RBCmd;
    nv = RBCnv;
    KL(dev::force, (k_cnf(nc*nv*md)), (md, nv, vert, adj0, adj1, nc, av, /**/ (float*)ff));
}

void apply(const Quants q, const TicketT t, /**/ Force *ff) {
    if (q.nc <= 0) return;
    area_volume(q.nc, t.texvert, t.textri, /**/ q.av);
    apply0(q.nc, t.texvert, t.texadj0, t.texadj1, q.av, /**/ ff);
}
