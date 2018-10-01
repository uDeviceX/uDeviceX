static void print(const char *s, float *dev) {
    float hst;
    cD2H(&hst, dev, 1);
    msg_print("%s: %g", s, hst);
}

static void dump(int n, float *dev) {
    int i;
    float *hst;
    EMALLOC(n, &hst);
    cD2H(hst, dev, n);
    for (i = 0; i < n; i++)
        printf("%g\n", hst[i]);
    EFREE(hst);
}

static void dump3(int n, float *dev) {
    int i, j;
    float *hst;
    EMALLOC(3*n, &hst);
    cD2H(hst, dev, 3*n);
    for (j = i = 0; i < n; i++) {
        printf("%g %g %g\n", hst[j], hst[j+1], hst[j+2]);
        j += 3;
    }
    EFREE(hst);
}

static void sum(int nv, int nc, const float *from, /**/ float *to) {
    dim3 thrd(128, 1);
    dim3 blck(ceiln(nv, thrd.x), nc);
    Dzero(to, nc);
    KL(juelicher_dev::sum, (blck, thrd), (nv, from, /**/ to));
}

void juelicher_apply(Juelicher *q, const RbcParams *par, const RbcQuants *quants, /**/ Force *ff) {
    int nc, ne, nt, nv;
    RbcParams_v parv;

    int4 *tri, *dih;
    float *area, *lentheta, *theta;
    float *lentheta_tot, *area_tot, *curva_mean_area_tot;
    float *f, *fad;
    const Particle *pp;

    float H0 = -1.0/2.0;
    float kb = 1.0;

    tri = q->tri; dih = q->dih;
    area = q->area; lentheta = q->lentheta; theta = q->theta;
    lentheta_tot = q->lentheta_tot;
    area_tot = q->area_tot;
    curva_mean_area_tot = q->curva_mean_area_tot;
    f = q->f;
    fad = q->fad;
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
    sum(nv, nc, area, /**/ area_tot); dSync();

    Dzero(theta, ne*nc);
    Dzero(lentheta, nv*nc);
    KL(juelicher_dev::compute_theta_len, (k_cnf(ne*nc)), (nv, ne, nc, pp, dih, /**/ theta, lentheta));
    dSync();

    sum(nv, nc, lentheta, /**/ lentheta_tot); dSync();
    KL(juelicher_dev::compute_mean_curv, (k_cnf(nc)), (nc, H0, kb, lentheta_tot, area_tot, /**/ curva_mean_area_tot));
    dSync();

    Dzero(f, nv*nc);
    Dzero(fad, nv*nc);
    KL(juelicher_dev::force_edg, (k_cnf(ne*nc)),
       (nv, ne, nc, H0, pp, dih, curva_mean_area_tot, theta, lentheta, area, /**/ f, fad));
    dSync();

    KL(juelicher_dev::force_lentheta, (k_cnf(ne*nc)),
       (nv, ne, nc, H0, pp, dih, curva_mean_area_tot, lentheta, area, /**/ f, fad));
    dSync();

    print("lentheta_tot", &lentheta_tot[0]);
    print("curva_mean_area_tot", &curva_mean_area_tot[0]);
    print("area", &area[0]);
    print("fx",  &f[0]);
    print("fad", &fad[0]);

    dump3(nv*nc, f);
}
