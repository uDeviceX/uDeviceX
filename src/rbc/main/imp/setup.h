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

    edg->a = a; edg->b = b; edg->c = c; edg->A = A;
}

static void setup_edg1(int md, int nv, int *adj0, int *adj1, float *rr, /**/ Edg *edg) {
    int valid, i;
    rbc::adj::Map m;

    for (i = 0; i < md*nv; i++) {
        valid = rbc::adj::hst(md, nv, i, adj0, adj1, /**/ &m);
        if (!valid) continue;
        setup_edg0(rr, m, /**/ &edg[i]);
        MSG("A: %g %g %g", edg[i].a, edg[i].b, edg[i].c);
    }
}

static void setup_edg(int md, int nv, int *adj0, int *adj1, /**/ Edg *dev) {
    float *rr;
    Edg *hst;
    hst = (Edg*) malloc(md*nv*sizeof(EdgInfo));
    rr = (float*)    malloc(3*nv*sizeof(float));

    evert("rbc.off", nv, /**/ rr);
    setup_edg1(md, nv, adj0, adj1, rr, /**/ hst);

    cH2D(dev, hst, md*nv);

    free(hst); free(rr);
}

static void setup(int md, int nt, int nv, const char *r_templ, /**/
                  Edg *edg, int4 *faces, int4 *tri, int *adj0, int *adj1) {
    int a0[nv*md], a1[nv*md];
    efaces(r_templ, nt, /**/ faces);
    rbc::adj::ini(md, nt, nv, faces, /**/ a0, a1);

    if (RBC_STRESS_FREE) setup_edg(md, nv, a0, a1, /**/ edg);
    cH2D(tri, faces, nt);
    cH2D(adj0, a0, nv*md);
    cH2D(adj1, a1, nv*md);
}
