static void random(int n, RbcRnd *rnd, /**/ float **r) {
    if (RBC_RND) {
        rbc_rnd_gen(rnd, n);
        *r = rnd->r;
    } else  {
        *r = NULL;
    }
}

static void apply0(int nc,
                   const Particle *pp, RbcRnd *rnd,
                   const int *adj0, const int *adj1, const Shape shape,
                   float *av, /**/ Force *ff){
    float *rnd0;    
    int md, nv;
    md = RBCmd; nv = RBCnv;
    random(nc * md * nv, rnd, /**/ &rnd0);
    KL(dev::force, (k_cnf(nc*nv*md)), (md, nv, nc, pp, rnd0,
                                       adj0, adj1, shape, av, /**/ (float*)ff));
}

/* temporary hack; TODO: remove this */
static void ini_rbc_params(RbcParams *p) {
    rbc_params_set_fluct(RBCgammaC, RBCkbT, p);
    rbc_params_set_bending(RBCkb, RBCphi, p);
    rbc_params_set_spring(RBCp, RBCx0, p);
    rbc_params_set_area_volume(RBCka, RBCkd, RBCkv, p);
}

void rbc_force_apply(const RbcQuants q, const RbcForce t, const RbcParams *p, /**/ Force *ff) {
    if (q.nc <= 0) return;
    area_volume::main(q.nt, q.nv, q.nc, q.pp, q.tri, /**/ q.av);
    apply0(q.nc, q.pp, t.rnd,
           q.adj0, q.adj1, q.shape, q.av, /**/ ff);
}
