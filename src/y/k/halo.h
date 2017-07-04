namespace k_halo {
__constant__ int cellpackstarts[27];
struct CellPackSOA {
    int *start, *count, *scan, size;
};
__constant__ CellPackSOA cellpacks[26];
struct SendBagInfo {
    int *start_src, *count_src, *start_dst;
    int bagsize, *scattered_entries;
    Particle *dbag, *hbag;
};

__constant__ SendBagInfo baginfos[26];
__constant__ int *srccells[26 * 2], *dstcells[26 * 2];

static __device__ int get_idpack(const int a[], const int i) {  /* where is `i' in sorted a[27]? */
  int k1, k3, k9;
  k9 = 9 * ((i >= a[9])           + (i >= a[18]));
  k3 = 3 * ((i >= a[k9 + 3])      + (i >= a[k9 + 6]));
  k1 =      (i >= a[k9 + k3 + 1]) + (i >= a[k9 + k3 + 2]);
  return k9 + k3 + k1;
}

/* returns halo box; 0 is a corner of subdomain */
static __device__ void get_box(int i, /**/ int org[3], int ext[3]) {
  /* i, org, ext : halo id, origin, extend */
  int L[3] = {XS, YS, ZS};
  int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
  int c;
  for (c = 0; c < 3; ++c) {
    org[c] = (d[c] == 1) ? L[c] - 1 : 0;
    ext[c] = (d[c] == 0) ? L[c]     : 1;
  }
}
  
__global__ void count_all(int *start, int *count) {
    int gid;
    int hid; /* halo id */
    int ndstcells, dstcid, srcentry, c;
    int org[3], ext[3]; /* halo [or]i[g]in and [ext]end */
    int srccellpos[3];
    
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= cellpackstarts[26]) return;

    hid = get_idpack(cellpackstarts, gid);
    get_box(hid, /**/ org, ext);
    ndstcells = ext[0] * ext[1] * ext[2];
    dstcid = gid - cellpackstarts[hid];

    if (dstcid < ndstcells) {
        int dstcellpos[3] = {dstcid % ext[0], (dstcid / ext[0]) % ext[1], dstcid / (ext[0] * ext[1])};
        for (c = 0; c < 3; ++c) srccellpos[c] = org[c] + dstcellpos[c];
        srcentry = srccellpos[0] + XS * (srccellpos[1] + YS * srccellpos[2]);
        cellpacks[hid].start[dstcid] = start[srcentry];
        cellpacks[hid].count[dstcid] = count[srcentry];
    } else if (dstcid == ndstcells) {
        cellpacks[hid].start[dstcid] = 0;
        cellpacks[hid].count[dstcid] = 0;
    }
}__global__ void copycells(int n) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    if (gid >= cellpackstarts[26]) return;

    int idpack = get_idpack(cellpackstarts, gid);
    int offset = gid - cellpackstarts[idpack];

    dstcells[idpack][offset] = srccells[idpack][offset];
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
    int idpack = get_idpack(cellpackstarts, gid);

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
