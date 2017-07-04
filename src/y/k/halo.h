namespace k_halo {
__constant__ int cellpackstarts[27];
struct CellPackSOA {
    int *start, *count, *scan, size;
    bool enabled;
};
__constant__ CellPackSOA cellpacks[26];
struct SendBagInfo {
    int *start_src, *count_src, *start_dst;
    int bagsize, *scattered_entries;
    Particle *dbag, *hbag;
};

__constant__ SendBagInfo baginfos[26];
__constant__ int *srccells[26 * 2], *dstcells[26 * 2];

__device__ int get_idpack(const int cellpackstarts[], const int gid) {  /* where is `i' in sorted a[27]? */
  int key9 = 9 * ((gid >= cellpackstarts[9]) + (gid >= cellpackstarts[18]));
  int key3 = 3 * ((gid >= cellpackstarts[key9 + 3]) + (gid >= cellpackstarts[key9 + 6]));
  int key1 = (gid >= cellpackstarts[key9 + key3 + 1]) + (gid >= cellpackstarts[key9 + key3 + 2]);
  return key9 + key3 + key1;
}
  
__global__ void count_all(int *cellsstart,
                          int *cellscount, int ntotalcells) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= cellpackstarts[26]) return;

    int key9 = 9 * ((gid >= cellpackstarts[9]) + (gid >= cellpackstarts[18]));
    int key3 = 3 * ((gid >= cellpackstarts[key9 + 3]) + (gid >= cellpackstarts[key9 + 6]));
    int key1 = (gid >= cellpackstarts[key9 + key3 + 1]) + (gid >= cellpackstarts[key9 + key3 + 2]);
    int idpack = key9 + key3 + key1;

    int d[3] = {(idpack + 2) % 3 - 1, (idpack / 3 + 2) % 3 - 1,
                (idpack / 9 + 2) % 3 - 1};
    int L[3] = {XS, YS, ZS};

    int halo_start[3];
    for (int c = 0; c < 3; ++c)
    halo_start[c] = max(d[c] * L[c] - L[c] / 2 - 1, -L[c] / 2);

    int halo_size[3];
    for (int c = 0; c < 3; ++c)
    halo_size[c] = min(d[c] * L[c] + L[c] / 2 + 1, L[c] / 2) - halo_start[c];

    int ndstcells = halo_size[0] * halo_size[1] * halo_size[2];
    int dstcid = gid - cellpackstarts[idpack];

    if (dstcid < ndstcells) {
        int dstcellpos[3] = {dstcid % halo_size[0],
                             (dstcid / halo_size[0]) % halo_size[1],
                             dstcid / (halo_size[0] * halo_size[1])};

        int srccellpos[3];
        for (int c = 0; c < 3; ++c)
        srccellpos[c] = halo_start[c] + dstcellpos[c] + L[c] / 2;

        int srcentry =
            srccellpos[0] +
            XS * (srccellpos[1] + YS * srccellpos[2]);
        int enabled = cellpacks[idpack].enabled;

        cellpacks[idpack].start[dstcid] = enabled * cellsstart[srcentry];
        cellpacks[idpack].count[dstcid] = enabled * cellscount[srcentry];
    } else if (dstcid == ndstcells) {
        cellpacks[idpack].start[dstcid] = 0;
        cellpacks[idpack].count[dstcid] = 0;
    }
}
template <int slot> __global__ void copycells(int n) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid >= cellpackstarts[26]) return;

    int key9 = 9 * ((gid >= cellpackstarts[9]) + (gid >= cellpackstarts[18]));
    int key3 = 3 * ((gid >= cellpackstarts[key9 + 3]) + (gid >= cellpackstarts[key9 + 6]));
    int key1 = (gid >= cellpackstarts[key9 + key3 + 1]) + (gid >= cellpackstarts[key9 + key3 + 2]);
    int idpack = key9 + key3 + key1;

    int offset = gid - cellpackstarts[idpack];

    dstcells[idpack + 26 * slot][offset] = srccells[idpack + 26 * slot][offset];
}

template <int NWARPS> __global__ void scan_diego() {
    __shared__ int shdata[32];

    int code = blockIdx.x;
    int *count = cellpacks[code].count;
    int *start = cellpacks[code].scan;
    int n = cellpacks[code].size;

    int tid = threadIdx.x;
    int laneid = threadIdx.x & 0x1f;
    int warpid = threadIdx.x >> 5;

    int lastval = 0;
    for (int sourcebase = 0; sourcebase < n; sourcebase += 32 * NWARPS) {
        int sourceid = sourcebase + tid;
        int mycount = 0, myscan = 0;
        if (sourceid < n) myscan = mycount = count[sourceid];
        if (tid == 0) myscan += lastval;

        for (int L = 1; L < 32; L <<= 1) {
            int val = __shfl_up(myscan, L);
            if (laneid >= L) myscan += val;
        }

        if (laneid == 31) shdata[warpid] = myscan;
        __syncthreads();
        if (warpid == 0) {
            int gs = 0;
            if (laneid < NWARPS) gs = shdata[tid];
            for (int L = 1; L < 32; L <<= 1) {
                int val = __shfl_up(gs, L);
                if (laneid >= L) gs += val;
            }

            shdata[tid] = gs;
            lastval = __shfl(gs, 31);
        }
        __syncthreads();
        if (warpid) myscan += shdata[warpid - 1];
        __syncthreads();
        if (sourceid < n) start[sourceid] = myscan - mycount;
    }
}
  
__global__ void fill_all(Particle *particles, int np,
                         int *required_bag_size) {
    int gid = (threadIdx.x >> 4) + 2 * blockIdx.x;
    if (gid >= cellpackstarts[26]) return;

    int key9 = 9 * ((gid >= cellpackstarts[9]) + (gid >= cellpackstarts[18]));
    int key3 = 3 * ((gid >= cellpackstarts[key9 + 3]) + (gid >= cellpackstarts[key9 + 6]));
    int key1 = (gid >= cellpackstarts[key9 + key3 + 1]) + (gid >= cellpackstarts[key9 + key3 + 2]);
    int idpack = key9 + key3 + key1;

    int cellid = gid - cellpackstarts[idpack];
    int tid = threadIdx.x & 0xf;
    int base_src = baginfos[idpack].start_src[cellid];
    int base_dst = baginfos[idpack].start_dst[cellid];
    int nsrc =
        min(baginfos[idpack].count_src[cellid], baginfos[idpack].bagsize - base_dst);
    int nfloats = nsrc * 6;
    for (int i = 2 * tid; i < nfloats; i += warpSize) {
        int lpid = i / 6;
        int dpid = base_dst + lpid;
        int spid = base_src + lpid;
        int c = i % 6;
        float2 word = *(float2 *)&particles[spid].r[c];
        *(float2 *)&baginfos[idpack].dbag[dpid].r[c] = word;
    }
    for (int lpid = tid; lpid < nsrc; lpid += warpSize / 2) {
        int dpid = base_dst + lpid;
        int spid = base_src + lpid;
        baginfos[idpack].scattered_entries[dpid] = spid;
    }
    if (gid + 1 == cellpackstarts[idpack + 1]) required_bag_size[idpack] = base_dst;
}
}
