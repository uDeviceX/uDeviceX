void rbc_bending_apply(RbcForce *t, const RbcParams *par, float /* dt */ , const RbcQuants *q, /**/ Force *ff) {
    int nc, nv, md;
    RbcParams_v parv;
    if (q->nc <= 0) return;
    if (!d::is_device_pointer(q->pp))  ERR("`q->pp` is not a device pointer");
    md = q->md; nv = q->nv; nc = q->nc;
    parv = rbc_params_get_view(par);
    KL(rbc_bending_dev::force, (k_cnf(q->nc*nv*md)), (parv, md, nv, nc, q->pp, *t->adj_v, /**/ (float*)ff));    
}
