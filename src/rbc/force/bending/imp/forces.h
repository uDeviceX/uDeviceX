static void apply(RbcParams_v parv, int nc, int nv, int md,
                  const Particle *pp,
                  const Adj_v *adj_v, /**/ Force *ff) {
    if (!d::is_device_pointer(ff))  ERR("`ff` is not a device pointer");
    KL(rbc_bending_dev::force, (k_cnf(nc*nv*md)), (parv, md, nv, nc, pp, *adj_v, /**/ (float*)ff));
}

static void dispatch_rnd(RbcParams_v parv, int nc, int nv, int md,
                         const Particle *pp,
                         const Adj_v *adj_v, RbcForce *t, /**/ Force *ff){
    apply(parv, nc, nv, md, pp, adj_v, /**/ ff);
}


void rbc_bending_apply(RbcForce *t, const RbcParams *par, float /* dt */ , const RbcQuants *q, /**/ Force *ff) {
    RbcParams_v parv;
    if (q->nc <= 0) return;
    if (!d::is_device_pointer(q->pp))  ERR("`q->pp` is not a device pointer");

    parv = rbc_params_get_view(par);
    dispatch_rnd(parv, q->nc, q->nv, q->md, q->pp, t->adj_v, t, /**/ ff);
}
