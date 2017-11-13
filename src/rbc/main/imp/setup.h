static void setup_edg(int md, int nv, int *a0, int *a1, /**/ EdgInfo *dev) {
    float *rr;
    EdgInfo *hst;
    hst = (EdgInfo*) malloc(md*nv*sizeof(EdgInfo));
    rr = (float*)    malloc(3*nv*sizeof(float));

    evert("rbc.off", nv, /**/ rr);
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
