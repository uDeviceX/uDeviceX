static void random(int n, rbc::rnd::D *rnd, /**/ float **r) {
    if (RBC_RND) {
        rbc::rnd::gen(rnd, n);
        *r = rnd->r;
    } else  {
        *r = NULL;
    }
}

static void apply0(int nc,
                   const Particle *pp, rbc::rnd::D *rnd,
                   const int *adj0, const int *adj1, const Shape shape,
                   float *av, /**/ Force *ff){
    float *rnd0;    
    int md, nv;
    md = RBCmd; nv = RBCnv;
    random(nc * md * nv, rnd, /**/ &rnd0);
    KL(dev::force, (k_cnf(nc*nv*md)), (md, nv, nc, pp, rnd0,
                                       adj0, adj1, shape, av, /**/ (float*)ff));
}

void rbcforce_apply(const RbcQuants q, const RbcForce t, /**/ Force *ff) {
    if (q.nc <= 0) return;
    area_volume::main(q.nt, q.nv, q.nc, q.pp, q.tri, /**/ q.av);
    apply0(q.nc, q.pp, t.rnd,
           q.adj0, q.adj1, q.shape, q.av, /**/ ff);
}
