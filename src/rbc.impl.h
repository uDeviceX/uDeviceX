namespace rbc {

#define MAX_CELLS_NUM 100000
#define md 7

void reg_edg(int f, int x, int y,  int* hx, int* hy) {
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

void setup_support(int *data, int *data2, int nentries) {
  setup_texture(k_rbc::texAdjVert, int);

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &k_rbc::texAdjVert, data,
		     &k_rbc::texAdjVert.channelDesc, sizeof(int) * nentries));

  setup_texture(k_rbc::texAdjVert2, int);
  CC(cudaBindTexture(&textureoffset, &k_rbc::texAdjVert2, data2,
		     &k_rbc::texAdjVert.channelDesc, sizeof(int) * nentries));
}

void setup(int* faces) {
  const char* r_templ = "rbc.off";
  off::f2faces(r_templ, faces);

  int   *trs4 = new int  [4 * RBCnt];
  for (int ifa = 0, i0 = 0, i1 = 0; ifa < RBCnt; ifa++) {
    trs4 [i0++] = faces[i1++]; trs4[i0++] = faces[i1++]; trs4[i0++] = faces[i1++];
    trs4 [i0++] = 0;
  }

  float *devtrs4;
  CC(cudaMalloc(&devtrs4,       RBCnt * 4 * sizeof(int)));
  CC(cudaMemcpy( devtrs4, trs4, RBCnt * 4 * sizeof(int), H2D));
  delete[] trs4;

  int hx[RBCnv*md], hy[RBCnv*md], a1[RBCnv*md], a2[RBCnv*md];
  int i;
  for (i = 0; i < RBCnv*md; i++) hx[i] = a1[i] = a2[i] = -1;

  for (int ifa = 0; ifa < RBCnt; ifa++) {
    i = 3*ifa;
    int f0 = faces[i++], f1 = faces[i++], f2 = faces[i++];
    reg_edg(f0, f1, f2,   hx, hy); /* register an edge */
    reg_edg(f1, f2, f0,   hx, hy);
    reg_edg(f2, f0, f1,   hx, hy);
  }
  for (i = 0; i < RBCnv; i++) gen_a12(i, hx, hy, /**/ a1, a2);

  int *ptr, *ptr2;
  CC(cudaMalloc(&ptr, sizeof(int) * RBCnv*md));
  CC(cudaMemcpy(ptr, a1, sizeof(int) * RBCnv*md, H2D));

  CC(cudaMalloc(&ptr2, sizeof(int) * RBCnv*md));
  CC(cudaMemcpy(ptr2, a2, sizeof(int) * RBCnv*md, H2D));

  setup_support(ptr, ptr2, RBCnv*md);

  setup_texture(k_rbc::texTriangles4, int4);
  setup_texture(k_rbc::texVertices, float2);

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &k_rbc::texTriangles4, devtrs4,
		     &k_rbc::texTriangles4.channelDesc,
		     RBCnt * 4 * sizeof(int)));
}

void forces(int nc, Particle *pp, Force *ff, float* host_av) {
  if (nc == 0) return;

  size_t textureoffset;
  CC(cudaBindTexture(&textureoffset, &k_rbc::texVertices,
		     (float2*)pp,
		     &k_rbc::texVertices.channelDesc,
		     nc * RBCnv * sizeof(float) * 6));

  dim3 avThreads(256, 1);
  dim3 avBlocks(1, nc);

  CC(cudaMemsetAsync(host_av, 0, nc * 2 * sizeof(float)));
  k_rbc::areaAndVolumeKernel<<<avBlocks, avThreads>>>(host_av);
  CC(cudaPeekAtLastError());

  k_rbc::fall_kernel<<<k_cnf(nc*RBCnv*md)>>>(nc, host_av, (float*)ff);
}

}
