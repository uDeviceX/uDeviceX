namespace KernelsContact {
__global__ void bulk_3tpp(float2 *particles, int np,
                          int ncellentries, int nsolutes,
                          float *acc, float seed,
                          int mysoluteid);

__global__ void halo(int nparticles_padded, int ncellentries,
                     int nsolutes, float seed);

void setup() {
  texCellsStart.channelDesc = cudaCreateChannelDesc<int>();
  texCellsStart.filterMode = cudaFilterModePoint;
  texCellsStart.mipmapFilterMode = cudaFilterModePoint;
  texCellsStart.normalized = 0;

  texCellEntries.channelDesc = cudaCreateChannelDesc<int>();
  texCellEntries.filterMode = cudaFilterModePoint;
  texCellEntries.mipmapFilterMode = cudaFilterModePoint;
  texCellEntries.normalized = 0;

  CC(cudaFuncSetCacheConfig(bulk_3tpp, cudaFuncCachePreferL1));
  CC(cudaFuncSetCacheConfig(halo, cudaFuncCachePreferL1));
}

__global__ void populate(uchar4 *subindices,
                         int *cellstart, int nparticles,
                         int soluteid, int ntotalparticles,
                         CellEntry *entrycells) {
#if !defined(__CUDA_ARCH__)
#warning __CUDA_ARCH__ not defined! assuming 350
#define _ACCESS(x) __ldg(x)
#elif __CUDA_ARCH__ >= 350
#define _ACCESS(x) __ldg(x)
#else
#define _ACCESS(x) (*(x))
#endif

  int warpid = threadIdx.x >> 5;
  int tid = threadIdx.x & 0x1f;

  int base = 32 * (warpid + 4 * blockIdx.x);
  int pid = base + tid;

  if (pid >= nparticles) return;

  uchar4 subindex = subindices[pid];

  if (subindex.x == 0xff && subindex.y == 0xff && subindex.z == 0xff) return;

  int cellid = subindex.x + XCELLS * (subindex.y + YCELLS * subindex.z);
  int mystart = _ACCESS(cellstart + cellid);
  int slot = mystart + subindex.w;

  CellEntry myentrycell;
  myentrycell.pid = pid;
  myentrycell.code.w = soluteid;

  entrycells[slot] = myentrycell;
}


void bind(const int *const cellsstart, const int *const cellentries,
          const int ncellentries, std::vector<ParticlesWrap> wsolutes,
          cudaStream_t stream, const int *const cellscount) {
  size_t textureoffset = 0;

  if (ncellentries)
    CC(cudaBindTexture(&textureoffset, &texCellEntries, cellentries,
                       &texCellEntries.channelDesc,
                       sizeof(int) * ncellentries));
  int ncells = XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN;
  CC(cudaBindTexture(&textureoffset, &texCellsStart, cellsstart,
                     &texCellsStart.channelDesc, sizeof(int) * ncells));
  int n = wsolutes.size();

  int ns[n];
  float2 *ps[n];
  float *as[n];

  for (int i = 0; i < n; ++i) {
    ns[i] = wsolutes[i].n;
    ps[i] = (float2 *)wsolutes[i].p;
    as[i] = (float *)wsolutes[i].a;
  }

  CC(cudaMemcpyToSymbolAsync(cnsolutes, ns, sizeof(int) * n, 0,
                             cudaMemcpyHostToDevice, stream));
  CC(cudaMemcpyToSymbolAsync(csolutes, ps, sizeof(float2 *) * n, 0,
                             cudaMemcpyHostToDevice, stream));
  CC(cudaMemcpyToSymbolAsync(csolutesacc, as, sizeof(float *) * n, 0,
                             cudaMemcpyHostToDevice, stream));
}
}
