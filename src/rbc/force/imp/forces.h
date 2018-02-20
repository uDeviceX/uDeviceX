static void get_stress_view(const RbcForce *f, /**/ StressFul_v *v) {
    *v = f->sinfo.sful;
}

static void get_stress_view(const RbcForce *f, /**/ StressFree_v *v) {
    *v = f->sinfo.sfree;
}

static void random(int n, RbcRnd *rnd, /**/ float **r) {
    if (RBC_RND) rbc_rnd_gen(rnd, n, /**/ r);
    else *r = NULL;
}

template <typename Stress_v>
static void apply(float dt, RbcParams_v parv, int nc, int nv,
                  const Particle *pp, RbcRnd *rnd,
                  const Adj_v *adj_v, const Shape shape, Stress_v sv,
                  float *av, /**/ Force *ff){
    if (!d::is_device_pointer(ff))  ERR("`ff` is not a device pointer");
    float *rnd0;
    int md;
    md = RBCmd;
    random(nc * md * nv, rnd, /**/ &rnd0);
    KL(rbc_force_dev::force, (k_cnf(nc*nv*md)), (dt, parv, md, nv, nc, pp, rnd0,
                                                 *adj_v, shape, sv, av, /**/ (float*)ff));
}

void rbc_force_apply(RbcForce *t, const RbcParams *par, float dt, const RbcQuants *q, /**/ Force *ff) {
    RbcParams_v parv;
    float *av;
    if (q->nc <= 0) return;
    if (!d::is_device_pointer(q->pp))  ERR("`q->pp` is not a device pointer");

    parv = rbc_params_get_view(par);
    UC(area_volume_compute(q->area_volume, q->nc, q->pp, /**/ &av));

    if (is_stress_free(t)) {
        StressFree_v si;
        get_stress_view(t, &si);
        apply(dt,
              parv, q->nc, q->nv, q->pp, t->rnd,
              t->adj_v, q->shape, si, av, /**/ ff);
    }
    else {
        StressFul_v si;
        get_stress_view(t, &si);
        apply(dt,
              parv, q->nc, q->nv, q->pp, t->rnd,
              t->adj_v, q->shape, si, av, /**/ ff);
    }    
}
