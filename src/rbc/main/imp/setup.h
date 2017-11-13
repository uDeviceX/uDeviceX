static void efaces(const char *f, int n0, /**/ int4 *faces) {
    /* get faces */
    int n;
    n = off::faces(f, n0, faces);
    if (n0 != n)
        ERR("wrong faces number in <%s> : %d != %d", f, n0, n);
}

static void evert(const char *f, int n0, /**/ float *vert) {
    /* get vertices */
    int n;
    n = off::vert(f, n0, vert);
    if (n0 != n)
        ERR("wrong vert number in <%s> : %d != %d", f, n0, n);
}

static void setup_edg(int md, int nt, int nv, int4 *faces, /**/ EdgInfo *edg) {
    float *rr;
    rr = (float*) malloc(3*nv*sizeof(float));
    evert("rbc.off", nv, /**/ rr);
    free(rr);
}

static void setup(int md, int nt, int nv, const char *r_templ, /**/
                  EdgInfo *edg, int4 *faces, int4 *tri, int *adj0, int *adj1) {
    int a1[nv*md], a2[nv*md];
    efaces(r_templ, nt, /**/ faces);
    rbc::adj::ini(md, nt, nv, faces, /**/ a1, a2);
    
    if (RBC_STRESS_FREE) setup_edg(md, nt, nv, faces, /**/ edg);

    cH2D(tri, faces, nt);
    cH2D(adj0, a1, nv*md);
    cH2D(adj1, a2, nv*md);
}
