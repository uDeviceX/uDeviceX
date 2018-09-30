static void dump(int n, float *dev) {
    int i;
    float *hst;
    EMALLOC(n, &hst);
    cD2H(hst, dev, n);
    for (i = 0; i < n; i++)
        printf("%g\n", hst[i]);
    EFREE(hst);
}

void juelicher_apply(Juelicher *q, const RbcParams *par, const RbcQuants *quants, /**/ Force *ff) {
    int nc, ne, nt, nv;
    RbcParams_v parv;

    int4 *tri, *dih;
    float *area, *lentheta, *theta;
    float *lentheta_tot, *area_tot;
    const Particle *pp;

    tri = q->tri; dih = q->dih;
    area = q->area; lentheta = q->lentheta;
    lentheta_tot = q->lentheta_tot;
    area_tot = q->area_tot;
    pp = quants->pp;

    nv = quants->nv; nc = quants->nc; nt = quants->nt;
    ne = q->ne;

    if (!d::is_device_pointer(pp))
        ERR("`q->pp` is not a device pointer");
    if (quants->nc <= 0) return;
    
    parv = rbc_params_get_view(par);

    Dzero(area, nv*nc);
    KL(juelicher_dev::compute_area, (k_cnf(nt*nc)), (nv, nt, nc, pp, tri, /**/ area));
    dSync();
    dump(nv*nc, area);
}
