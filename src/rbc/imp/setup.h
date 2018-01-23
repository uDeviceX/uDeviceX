static void setup_edg0(float *rr, AdjMap m, /**/ Edg *edg) {
    int i0, i1, i2;
    float *r0, *r1, *r2;
    float r01[3], r12[3], r20[3];
    float a, b, c, A; /* edges and area */

    i0 = m.i0; i1 = m.i1; i2 = m.i2;

    r0 = &rr[3*i0]; r1 = &rr[3*i1]; r2 = &rr[3*i2];

    diff(r0, r1, /**/ r01);
    diff(r1, r2, /**/ r12);
    diff(r2, r0, /**/ r20);

    a = vabs(r01); b = vabs(r12); c = vabs(r20);
    A = area_heron(a, b, c);

    edg->a = a; edg->A = A;
}

static void setup_edg1(int md, int nv, Adj *adj, float *rr, /**/
                       Edg *edg, float *ptotArea) {
    int valid, i;
    AdjMap m;
    float totArea;

    totArea = 0;
    for (i = 0; i < md*nv; i++) {
        valid = adj_get_map(md, nv, i, adj, /**/ &m);
        if (!valid) continue;
        setup_edg0(rr, m, /**/ &edg[i]);
        totArea += edg[i].A;
    }
    totArea /= 3; /* seen every face three times */

    msg_print("totArea: %g", totArea);
    *ptotArea = totArea;
}

static void setup_edg(int md, int nv, Adj *adj, /**/ Edg *dev, float *totArea) {
    float *rr;
    Edg *hst;
    const char *fn = "rbc.stress.free";

    UC(emalloc(md*nv*sizeof(Edg), (void**) &hst));
    UC(emalloc(3*nv*sizeof(float), (void**) &rr));
    msg_print("reading <%s>", fn);
    UC(evert(fn, nv, /**/ rr));
    UC(setup_edg1(md, nv, adj, rr, /**/ hst, totArea));

    cH2D(dev, hst, md*nv);

    UC(efree(hst)); UC(efree(rr));
}

static void setup_anti(int md, int nv, Adj *adj, /**/ int *dev) {
    int n;
    int *hst;
    n = md*nv;
    UC(emalloc(n*sizeof(int), (void**) &hst));

    adj_get_anti(md, nv, adj, /**/ hst);
    cH2D(dev, hst, n);

    free(hst);
}

static void setup0(int md, int nt, int nv, int4 *faces, /**/
                   int *anti, Edg *edg, float *totArea, int *adj0, int *adj1) {
    Adj adj;
    adj_ini(md, nt, nv, faces, /**/ &adj);

    if (RBC_STRESS_FREE) UC(setup_edg(md,  nv, &adj, /**/ edg, totArea));
    if (RBC_RND)         UC(setup_anti(md, nv, &adj, /**/ anti));

    cH2D(adj0, adj.adj0, nv*md); /* TODO */
    cH2D(adj1, adj.adj1, nv*md);

    adj_fin(&adj);
}

static void setup(int md, int nt, int nv, const char *cell, /**/
                  int *anti, Edg *edg, float *totArea, int4 *faces, AreaVolume *area_volume,
                  int *adj0, int *adj1) {
    UC(efaces(cell, nt, /**/ faces));
    UC(area_volume_setup(nt, nv, faces, /**/ area_volume));
    UC(setup0(md, nt, nv, faces, /**/ anti, edg, totArea, adj0, adj1));
}
