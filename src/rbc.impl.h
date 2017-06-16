namespace rbc
{
#define MAX_CELLS_NUM 100000
#define md 7

void reg(int f, int x, int y,  int* hx, int* hy) {
    int j = f*md;
    while (hx[j] != -1) j++;
    hx[j] = x; hy[j] = y;
}

int nxt(int i, int x,   int* hx, int* hy) {
    i *= md;
    while (hx[i] != x) i++;
    return hy[i];
}

void gen_a12(int i0, int* hx, int* hy, /**/ int* a1, int* a2) {
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

    float *devtrs4;
    CC(cudaMalloc(&devtrs4,       RBCnt * 4 * sizeof(int)));
    cH2D(devtrs4, trs4, RBCnt * 4);
    delete[] trs4;

    int hx[RBCnv*md], hy[RBCnv*md], a1[RBCnv*md], a2[RBCnv*md];
    int i;
    for (i = 0; i < RBCnv*md; i++) hx[i] = a1[i] = a2[i] = -1;

    for (int ifa = 0; ifa < RBCnt; ifa++) {
        i = 3*ifa;
        int f0 = faces[i++], f1 = faces[i++], f2 = faces[i++];
        reg(f0, f1, f2,   hx, hy); /* register an edge */
        reg(f1, f2, f0,   hx, hy);
        reg(f2, f0, f1,   hx, hy);
    }
    for (i = 0; i < RBCnv; i++) gen_a12(i, hx, hy, /**/ a1, a2);

    int *adj0, *adj1;
    CC(cudaMalloc(&adj0, sizeof(int) * RBCnv*md));
    cH2D(adj0, a1, RBCnv*md);

    CC(cudaMalloc(&adj1, sizeof(int) * RBCnv*md));
    cH2D(adj1, a2, RBCnv*md);

    setup_texture(k_rbc::Vert, float2);

    /* TODO free these arrays */
    /* TODO free the texobjs  */
    texadj0.setup(adj0, RBCnv*md);
    texadj1.setup(adj1, RBCnv*md);
    textri.setup((int4*)devtrs4, RBCnt);
}

void forces(int nc, Particle *pp, Force *ff, float* host_av) {
    if (nc <= 0) return;

    size_t offset;
    CC(cudaBindTexture(&offset, &k_rbc::Vert,
                       (float2*)pp,
                       &k_rbc::Vert.channelDesc,
                       nc * RBCnv * sizeof(float) * 6));

    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);

    CC(cudaMemsetAsync(host_av, 0, nc * 2 * sizeof(float)));
    k_rbc::area_volume<<<avBlocks, avThreads>>>(textri, host_av);
    CC(cudaPeekAtLastError());

    k_rbc::force<<<k_cnf(nc*RBCnv*md)>>>(texadj0, texadj1, nc, host_av, (float*)ff);
}
}
