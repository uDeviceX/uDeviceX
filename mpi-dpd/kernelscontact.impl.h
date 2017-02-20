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

__global__ void bulk_3tpp(float2 *particles, int np,
                          int ncellentries, int nsolutes,
                          float *acc, float seed,
                          int mysoluteid) {
  int gid = threadIdx.x + blockDim.x * blockIdx.x;
  int pid = gid / 3;
  int zplane = gid % 3;

  if (pid >= np) return;

  float2 dst0 = _ACCESS(particles + 3 * pid + 0);
  float2 dst1 = _ACCESS(particles + 3 * pid + 1);
  float2 dst2 = _ACCESS(particles + 3 * pid + 2);

  int scan1, scan2, ncandidates, spidbase;
  int deltaspid1, deltaspid2;

  {
    int xcenter = min(XCELLS - 1, max(0, XOFFSET + (int)floorf(dst0.x)));
    int xstart = max(0, xcenter - 1);
    int xcount = min(XCELLS, xcenter + 2) - xstart;

    if (xcenter - 1 >= XCELLS || xcenter + 2 <= 0) return;

    int ycenter = min(YCELLS - 1, max(0, YOFFSET + (int)floorf(dst0.y)));

    int zcenter = min(ZCELLS - 1, max(0, ZOFFSET + (int)floorf(dst1.x)));
    int zmy = zcenter - 1 + zplane;
    bool zvalid = zmy >= 0 && zmy < ZCELLS;

    int count0 = 0, count1 = 0, count2 = 0;

    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YCELLS) {
      int cid0 = xstart + XCELLS * (ycenter - 1 + YCELLS * zmy);
      spidbase = tex1Dfetch(texCellsStart, cid0);
      count0 = tex1Dfetch(texCellsStart, cid0 + xcount) - spidbase;
    }

    if (zvalid && ycenter >= 0 && ycenter < YCELLS) {
      int cid1 = xstart + XCELLS * (ycenter + YCELLS * zmy);
      deltaspid1 = tex1Dfetch(texCellsStart, cid1);
      count1 = tex1Dfetch(texCellsStart, cid1 + xcount) - deltaspid1;
    }

    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YCELLS) {
      int cid2 = xstart + XCELLS * (ycenter + 1 + YCELLS * zmy);
      deltaspid2 = tex1Dfetch(texCellsStart, cid2);
      count2 = tex1Dfetch(texCellsStart, cid2 + xcount) - deltaspid2;
    }

    scan1 = count0;
    scan2 = count0 + count1;
    ncandidates = scan2 + count2;

    deltaspid1 -= scan1;
    deltaspid2 -= scan2;
  }

  float xforce = 0, yforce = 0, zforce = 0;

#pragma unroll 3
  for (int i = 0; i < ncandidates; ++i) {
    int m1 = (int)(i >= scan1);
    int m2 = (int)(i >= scan2);
    int slot = i + (m2 ? deltaspid2 : m1 ? deltaspid1 : spidbase);

    CellEntry ce;
    ce.pid = tex1Dfetch(texCellEntries, slot);
    int soluteid = ce.code.w;

    ce.code.w = 0;

    int spid = ce.pid;

    if (mysoluteid < soluteid || mysoluteid == soluteid && pid <= spid)
      continue;

    int sentry = 3 * spid;
    float2 stmp0 = _ACCESS(csolutes[soluteid] + sentry);
    float2 stmp1 = _ACCESS(csolutes[soluteid] + sentry + 1);
    float2 stmp2 = _ACCESS(csolutes[soluteid] + sentry + 2);

    float myrandnr = Logistic::mean0var1(seed, pid, spid);

    // check for particle types and compute the DPD force
    float3 pos1 = make_float3(dst0.x, dst0.y, dst1.x),
           pos2 = make_float3(stmp0.x, stmp0.y, stmp1.x);
    float3 vel1 = make_float3(dst1.y, dst2.x, dst2.y),
           vel2 = make_float3(stmp1.y, stmp2.x, stmp2.y);
    int type1 = MEMB_TYPE;
    int type2 = MEMB_TYPE;
    float3 strength = compute_dpd_force_traced(type1, type2, pos1, pos2,
                                                     vel1, vel2, myrandnr);

    float xinteraction = strength.x;
    float yinteraction = strength.y;
    float zinteraction = strength.z;

    xforce += xinteraction;
    yforce += yinteraction;
    zforce += zinteraction;

    atomicAdd(csolutesacc[soluteid] + sentry, -xinteraction);
    atomicAdd(csolutesacc[soluteid] + sentry + 1, -yinteraction);
    atomicAdd(csolutesacc[soluteid] + sentry + 2, -zinteraction);
  }

  atomicAdd(acc + 3 * pid + 0, xforce);
  atomicAdd(acc + 3 * pid + 1, yforce);
  atomicAdd(acc + 3 * pid + 2, zforce);
}
}
