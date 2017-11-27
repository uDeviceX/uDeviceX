static void random(int n, rbc::rnd::D* rnd, /**/ float **r) {
    if (RBC_RND) {
        rbc::rnd::gen(rnd, n);
        *r = rnd->r;
    } else  {
        *r = NULL;
    }
}

static void apply0(int nc,
                   const Texo<float2> vert, rbc::rnd::D *rnd,
                   const int *adj0, const int *adj1, const Shape shape,
                   float* av, /**/ Force *ff){
    float *rnd0;    
    int md, nv;
    md = RBCmd; nv = RBCnv;
    random(nc * md * nv, rnd, /**/ &rnd0);
    KL(dev::force, (k_cnf(nc*nv*md)), (md, nv, nc, vert, rnd0,
                                       adj0, adj1, shape, av, /**/ (float*)ff));
}

void apply(const Quants q, const TicketT t, /**/ Force *ff) {
    if (q.nc <= 0) return;
    area_volume::dev(q.nt, q.nv, q.nc, q.pp, q.tri, /**/ q.av);
    apply0(q.nc, t.texvert, t.rnd,
           q.adj0, q.adj1, q.shape, q.av, /**/ ff);
}
