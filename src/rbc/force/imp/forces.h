static void random(int n, RbcRnd *rnd, /**/ float **r) {
    if (RBC_RND) rbc_rnd_gen(rnd, n, /**/ r);
    else *r = NULL;
}

static void apply0(float dt, RbcParams_v parv, int nc, int nv,
                   const Particle *pp, RbcRnd *rnd,
                   Adj_v *adj_v, const Shape shape,
                   float *av, /**/ Force *ff){
    float *rnd0;
    int md;
    md = RBCmd;
    random(nc * md * nv, rnd, /**/ &rnd0);
    KL(dev::force, (k_cnf(nc*nv*md)), (dt, parv, md, nv, nc, pp, rnd0,
                                       *adj_v, shape, av, /**/ (float*)ff));
}

void rbc_force_apply(float dt, const RbcQuants q, const RbcForce t, const RbcParams *par, /**/ Force *ff) {
    RbcParams_v parv;
    float *av;
    if (q.nc <= 0) return;
    parv = rbc_params_get_view(par);
    UC(area_volume_compute(q.area_volume, q.nc, q.pp, /**/ &av));
    apply0(dt,
           parv, q.nc, q.nv, q.pp, t.rnd,
           q.adj_v, q.shape, av, /**/ ff);
}
