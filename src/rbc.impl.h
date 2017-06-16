namespace rbc
{
/* [m]aximumd [d]egree, number of vertices, number of triangles */
#define md ( RBCmd )
#define nv ( RBCnv )
#define nt ( RBCnt )

static void reg(int f, int x, int y,  /**/ int *hx, int *hy) { /* register an edge */
    int j = f*md;
    while (hx[j] != -1) j++;
    hx[j] = x; hy[j] = y;
}

static int nxt(int i, int x, int *hx, int *hy) { /* next */
    i *= md;
    while (hx[i] != x) i++;
    return hy[i];
}

static void gen_a12(int i0, int *hx, int *hy, /**/ int *a1, int *a2) {
    int lo = i0*md, hi = lo + md, mi = hx[lo];
    int i;
    for (i = lo + 1; (i < hi) && (hx[i] != -1); i++)
    if (hx[i] < mi) mi = hx[i]; /* minimum */

    int c = mi, c0;
    i = lo;
    do {
	c     = nxt(i0, c0 = c, hx, hy);
	a1[i] = c0;
	a2[i] = nxt(c, c0, hx, hy);
	i++;
    }  while (c != mi);
}

void setup(int *faces) {
    const char r_templ[] = "rbc.off";
    off::f2faces(r_templ, faces);

    int   *trs4 = new int  [4 * nt];
    for (int ifa = 0, i0 = 0, i1 = 0; ifa < nt; ifa++) {
	trs4 [i0++] = faces[i1++]; trs4[i0++] = faces[i1++]; trs4[i0++] = faces[i1++];
	trs4 [i0++] = 0;
    }

    CC(cudaMalloc(&tri, nt * sizeof(int4)));
    cH2D(tri, (int4*) trs4, nt);
    delete[] trs4;

    int hx[nv*md], hy[nv*md], a1[nv*md], a2[nv*md];
    int i;
    for (i = 0; i < nv*md; i++) hx[i] = a1[i] = a2[i] = -1;

    for (int ifa = 0; ifa < nt; ifa++) {
	i = 3*ifa;
	int f0 = faces[i++], f1 = faces[i++], f2 = faces[i++];
	reg(f0, f1, f2,   hx, hy); /* register an edge */
	reg(f1, f2, f0,   hx, hy);
	reg(f2, f0, f1,   hx, hy);
    }
    for (i = 0; i < nv; i++) gen_a12(i, hx, hy, /**/ a1, a2);

    CC(cudaMalloc(&adj0, sizeof(int) * nv*md));
    cH2D(adj0, a1, nv*md);

    CC(cudaMalloc(&adj1, sizeof(int) * nv*md));
    cH2D(adj1, a2, nv*md);

    /* TODO free these arrays */
    /* TODO free the texobjs  */
    texadj0.setup(adj0, nv*md);
    texadj1.setup(adj1, nv*md);
    textri.setup(tri,   nt);
}

void forces(int nc, Particle *pp, Force *ff, float* host_av) {
    if (nc <= 0) return;

    /* TODO do this only once (need QuantsTickets for this) */
    texvert.setup((float2*) pp, 3*nc*nv);

    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);

    CC(cudaMemsetAsync(host_av, 0, nc * 2 * sizeof(float)));
    k_rbc::area_volume<<<avBlocks, avThreads>>>(texvert, textri, host_av);
    CC(cudaPeekAtLastError());

    k_rbc::force<<<k_cnf(nc*nv*md)>>>(texvert, texadj0, texadj1, nc, host_av, (float*)ff);

    /* TODO do this only once (need QuantsTickets for this) */
    dSync();
    texvert.destroy();
}

#undef md
#undef nv
#undef nt
}
