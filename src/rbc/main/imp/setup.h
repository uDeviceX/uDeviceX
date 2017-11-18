static void setup_edg0(float *rr, rbc::adj::Map m, /**/ Edg *edg) {
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
    A = heron(a, b, c);

    edg->a = a; edg->A = A;
}

static void setup_edg1(int md, int nv, rbc::adj::Hst *Adj, float *rr, /**/
                         Edg *edg, float *ptotArea) {
    int valid, i;
    rbc::adj::Map m;
    float totArea;

    totArea = 0;
    for (i = 0; i < md*nv; i++) {
        valid = rbc::adj::hst(md, nv, i, Adj, /**/ &m);
        if (!valid) continue;
        setup_edg0(rr, m, /**/ &edg[i]);
        totArea += edg[i].A;
    }
    totArea /= 3; /* seen every face three times */

    MSG("totArea: %g", totArea);
    *ptotArea = totArea;
}

static void setup_edg(int md, int nv, rbc::adj::Hst *Adj, /**/ Edg *dev, float *totArea) {
    float *rr;
    Edg *hst;
    hst = (Edg*) malloc(md*nv*sizeof(Edg));
    rr = (float*)malloc(3*nv*sizeof(float));

    evert("rbc.off", nv, /**/ rr);
    setup_edg1(md, nv, Adj, rr, /**/ hst, totArea);

    cH2D(dev, hst, md*nv);

    free(hst); free(rr);
}

static void setup0(int md, int nt, int nv, int4 *faces, /**/
                   Edg *edg, float *totArea, int *adj0, int *adj1) {
    rbc::adj::Hst Adj;
    rbc::adj::ini(md, nt, nv, faces, /**/ &Adj);

    if (RBC_STRESS_FREE) setup_edg(md, nv, &Adj, /**/ edg, totArea);
    cH2D(adj0, Adj.adj0, nv*md); /* TODO */
    cH2D(adj1, Adj.adj1, nv*md);

    rbc::adj::fin(&Adj);
}

static void setup(int md, int nt, int nv, const char *r_templ, /**/
                  Edg *edg, float *totArea, int4 *faces, int4 *tri, int *adj0, int *adj1) {
    efaces(r_templ, nt, /**/ faces);
    setup0(md, nt, nv, faces, /**/ edg, totArea, adj0, adj1);
    cH2D(tri, faces, nt);
}
