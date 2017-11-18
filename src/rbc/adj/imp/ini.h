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

void ini0(int md, int nt, int nv, int4 *faces, /**/ int *a1, int *a2) {
    int hx[nv*md], hy[nv*md];
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
}

static void alloc(int n, Hst *A) {
    A->adj0 = (int*)malloc(n*sizeof(int));
    A->adj1 = (int*)malloc(n*sizeof(int));
}

void ini(int md, int nt, int nv, int4 *faces, /**/ Hst *A) {
    int *a1, *a2;
    alloc(nv*nt, /**/ A);
    a1 = A->adj0; /* sic */
    a2 = A->adj1;
    ini0(md, nt, nv, faces, /**/ a1, a2);
}
