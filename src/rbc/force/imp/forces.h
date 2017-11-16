static void apply0(int nc,
                   const Texo<float2> vert, rbc::rnd::D *rnd,
                   const Texo<int> adj0, const Texo<int> adj1, const Shape shape,
                   float* av, /**/ Force *ff){
    int md, nv;
    md = RBCmd;
    nv = RBCnv;
    /* TODO */
    float *rnd0 = NULL;
    if (RBC_RND) rnd0 = rnd->r;
    KL(dev::force, (k_cnf(nc*nv*md)), (md, nv, nc, vert, rnd0,
                                       adj0, adj1, shape, av, /**/ (float*)ff));
}

void apply(const Quants q, const TicketT t, /**/ Force *ff) {
    if (q.nc <= 0) return;
    area_volume(q.nc, t.texvert, t.textri, /**/ q.av);
    apply0(q.nc, t.texvert, t.rnd,
           t.texadj0, t.texadj1, q.shape, q.av, /**/ ff);
}
