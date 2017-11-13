static void gfaces(const char *f, int n0, /**/ int4 *faces) {
    /* get faces */
    int n;
    n = off::faces(f, n0, faces);
    if (n0 != n)
        ERR("wrong faces number in <%s> : %d != %d", f, n0, n);
}
static void setup(int md, int nt, int nv, const char *r_templ, /**/ int4 *faces, int4 *tri, int *adj0, int *adj1) {
    gfaces(r_templ, nt, /**/ faces);

    int a1[nv*md], a2[nv*md];
    rbc::adj::ini(md, nt, nv, faces, /**/ a1, a2);

    cH2D(tri, faces, nt);
    cH2D(adj0, a1, nv*md);
    cH2D(adj1, a2, nv*md);
}
