void kantor_apply(Kantor *t, const RbcParams *par, const RbcQuants *q, /**/ Force *ff) {
    int nc, nv, md;
    RbcParams_v parv;
    if (q->nc <= 0) return;
    if (!d::is_device_pointer(q->pp))
        ERR("`q->pp` is not a device pointer");
    if (!d::is_device_pointer(ff))
        ERR("`ff` is not a device pointer");
    md = q->md; nv = q->nv; nc = q->nc;
    parv = rbc_params_get_view(par);
    KL(kantor_dev::force, (k_cnf(nc*nv*md)),
       (parv, md, nv, nc, q->pp, *t->adj_v, /**/ (float*)ff));
}
