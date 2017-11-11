static void reg(int md, int f, int x, int y,  /**/ int *hx, int *hy) { /* register an edge */
    int j = f*md;
    while (hx[j] != -1) j++;
    hx[j] = x; hy[j] = y;
}

static int nxt(int md, int i, int x, int *hx, int *hy) { /* next */
    i *= md;
    while (hx[i] != x) i++;
    return hy[i];
}

static void gen_a12(int md, int i0, int *hx, int *hy, /**/ int *a1, int *a2) {
    int lo = i0*md, hi = lo + md, mi = hx[lo];
    int i;
    for (i = lo + 1; (i < hi) && (hx[i] != -1); i++)
        if (hx[i] < mi) mi = hx[i]; /* minimum */

    int c = mi, c0;
    i = lo;
    do {
        c     = nxt(md, i0, c0 = c, hx, hy);
        a1[i] = c0;
        a2[i] = nxt(md, c, c0, hx, hy);
        i++;
    }  while (c != mi);
}

static void gfaces(const char *f, int n0, /**/ int4 *faces) {
    /* get faces */
    int n;
    n = off::faces(f, faces);
    if (n0 != n)
        ERR("wrong faces number in <%s> : %d != %d", f, n0, n);
}
static void setup(int md, int nt, int nv, const char *r_templ, int4 *faces, int4 *tri, int *adj0, int *adj1) {
    gfaces(r_templ, nt, /**/ faces);

    cH2D(tri, faces, nt);

    int hx[nv*md], hy[nv*md], a1[nv*md], a2[nv*md];
    int i;
    for (i = 0; i < nv*md; i++) hx[i] = a1[i] = a2[i] = -1;

    int4 t;
    for (int ifa = 0; ifa < nt; ifa++) {
        t = faces[ifa];
        int f0 = t.x, f1 = t.y, f2 = t.z;
        reg(md, f0, f1, f2,   hx, hy); /* register an edge */
        reg(md, f1, f2, f0,   hx, hy);
        reg(md, f2, f0, f1,   hx, hy);
    }
    for (i = 0; i < nv; i++) gen_a12(md, i, hx, hy, /**/ a1, a2);
    
    cH2D(adj0, a1, nv*md);
    cH2D(adj1, a2, nv*md);
}
