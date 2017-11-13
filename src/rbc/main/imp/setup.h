static void setup_edg0(rbc::adj::Map m, /**/ EdgInfo *edg) {
    int i0, i1, i2;
    i0 = m.i0; i1 = m.i1; i2 = m.i2;
    MSG("i012: %d %d %d", i0, i1, i2);
}

static void setup_edg1(int md, int nv, int *adj0, int *adj1, float *rr, /**/ EdgInfo *edg) {
    int valid, i;
    rbc::adj::Map m;

    for (i = 0; i < md*nv; i++) {
        valid = rbc::adj::hst(md, nv, i, adj0, adj1, /**/ &m);
        if (!valid) continue;
        setup_edg0(m, /**/ &edg[i]);
    }
}

static void setup_edg(int md, int nv, int *adj0, int *adj1, /**/ EdgInfo *dev) {
    float *rr;
    EdgInfo *hst;
    hst = (EdgInfo*) malloc(md*nv*sizeof(EdgInfo));
    rr = (float*)    malloc(3*nv*sizeof(float));

    evert("rbc.off", nv, /**/ rr);
    setup_edg1(md, nv, adj0, adj1, rr, /**/ hst);

    cH2D(dev, hst, md*nv);

    free(hst); free(rr);
}

static void setup(int md, int nt, int nv, const char *r_templ, /**/
                  EdgInfo *edg, int4 *faces, int4 *tri, int *adj0, int *adj1) {
    int a0[nv*md], a1[nv*md];
    efaces(r_templ, nt, /**/ faces);
    rbc::adj::ini(md, nt, nv, faces, /**/ a0, a1);

    if (RBC_STRESS_FREE) setup_edg(md, nv, a0, a1, /**/ edg);
    cH2D(tri, faces, nt);
    cH2D(adj0, a0, nv*md);
    cH2D(adj1, a1, nv*md);
}
