void juelicher_apply(Juelicher *t, const RbcParams *par, const RbcQuants *q, /**/ Force *ff) {
    int nc, ne, nt, nv;
    RbcParams_v parv;
    if (q->nc <= 0) return;
    if (!d::is_device_pointer(q->pp))
        ERR("`q->pp` is not a device pointer");
    nv = q->nv; nc = q->nc; nt = q->nt; ne = t->ne;
    parv = rbc_params_get_view(par);
    //    KL(juelicher_dev::force, (k_cnf(nc*nv*md)),
    //       (parv, md, nv, nc, q->pp, *t->adj_v, /**/ (float*)ff));
    KL(juelicher_dev::force2, (k_cnf(ne*nc)),
       (parv, ne, nc, q->pp, t->dih, /**/ (float*)ff));
}
