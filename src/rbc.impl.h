namespace rbc
{
void reg(int f, int x, int y,  int* hx, int* hy) {
    int j = f*RBCmd;
    while (hx[j] != -1) j++;
    hx[j] = x; hy[j] = y;
}

int nxt(int i, int x,   int* hx, int* hy) {
    i *= RBCmd;
    while (hx[i] != x) i++;
    return hy[i];
}

void gen_a12(int i0, int* hx, int* hy, /**/ int* a1, int* a2) {
    int md = RBCmd;
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

void setup(int* faces) {
    const char r_templ[] = "rbc.off";
    off::f2faces(r_templ, faces);

    int   *trs4 = new int  [4 * RBCnt];
    for (int ifa = 0, i0 = 0, i1 = 0; ifa < RBCnt; ifa++) {
        trs4 [i0++] = faces[i1++]; trs4[i0++] = faces[i1++]; trs4[i0++] = faces[i1++];
        trs4 [i0++] = 0;
    }

    CC(cudaMalloc(&tri, RBCnt * sizeof(int4)));
    cH2D(tri, (int4*) trs4, RBCnt);
    delete[] trs4;

    int hx[RBCnv*RBCmd], hy[RBCnv*RBCmd], a1[RBCnv*RBCmd], a2[RBCnv*RBCmd];
    int i;
    for (i = 0; i < RBCnv*RBCmd; i++) hx[i] = a1[i] = a2[i] = -1;

    for (int ifa = 0; ifa < RBCnt; ifa++) {
        i = 3*ifa;
        int f0 = faces[i++], f1 = faces[i++], f2 = faces[i++];
        reg(f0, f1, f2,   hx, hy); /* register an edge */
        reg(f1, f2, f0,   hx, hy);
        reg(f2, f0, f1,   hx, hy);
    }
    for (i = 0; i < RBCnv; i++) gen_a12(i, hx, hy, /**/ a1, a2);
    
    CC(cudaMalloc(&adj0, sizeof(int) * RBCnv*RBCmd));
    cH2D(adj0, a1, RBCnv*RBCmd);

    CC(cudaMalloc(&adj1, sizeof(int) * RBCnv*RBCmd));
    cH2D(adj1, a2, RBCnv*RBCmd);

    /* TODO free these arrays */
    /* TODO free the texobjs  */
    texadj0.setup(adj0, RBCnv*RBCmd);
    texadj1.setup(adj1, RBCnv*RBCmd);
    textri.setup(tri,   RBCnt);
}

void forces(int nc, Particle *pp, Force *ff, float* host_av) {
    if (nc <= 0) return;

    /* TODO do this only once (need QuantsTickets for this) */
    texvert.setup((float2*) pp, 3*nc*RBCnv);

    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);

    CC(cudaMemsetAsync(host_av, 0, nc * 2 * sizeof(float)));
    k_rbc::area_volume<<<avBlocks, avThreads>>>(texvert, textri, host_av);
    CC(cudaPeekAtLastError());

    k_rbc::force<<<k_cnf(nc*RBCnv*md)>>>(texvert, texadj0, texadj1, nc, host_av, (float*)ff);

    /* TODO do this only once (need QuantsTickets for this) */
    dSync();
    texvert.destroy();
}
}
