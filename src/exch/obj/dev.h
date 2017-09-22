namespace dev {

enum { MAX_DSTS = 7 };

static __device__ int map_code(int3 L, const float r[3]) {
    int x, y, z;
    enum {X, Y, Z};
    x = -1 + (r[X] >= -L.x / 2) + (r[X] >= L.x / 2);
    y = -1 + (r[Y] >= -L.y / 2) + (r[Y] >= L.y / 2);
    z = -1 + (r[Z] >= -L.z / 2) + (r[Z] >= L.z / 2);

    return frag_d2i(x, y, z);
}

static __device__ void add_to_map(int soluteid, int pid, int fid, Map m) {
    int ientry, centry;
    centry = soluteid * NBAGS + fid;
    ientry = atomicAdd(m.counts + centry, 1);
    m.ids[fid][ientry] = pid;
}

static __device__ int add_faces(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    for (int c = 0; c < 3; ++c) {
        if (d[c]) {
            int df[3] = {0, 0, 0}; df[c] = d[c];
            fids[j++] = frag_d32i(df);
        }
    }
    return j;
}

static __device__ int add_edges(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    enum {X, Y, Z};
    for (int c = 0; c < 3; ++c) {
        int de[3] = {d[X], d[Y], d[Z]}; de[c] = 0;
        if (de[(c + 1) % 3] && de[(c + 2) % 3])
            fids[j++] = frag_d32i(de);
    }
    return j;
}

static __device__ int add_cornr(int j, const int d[3], /**/ int fids[MAX_DSTS]) {
    enum {X, Y, Z};
    if (d[X] && d[Y] && d[Z])
        fids[j++] = frag_d32i(d);
    return j;
}

static __device__ int map_decode(int code, /**/ int fids[MAX_DSTS]) {
    int j = 0;
    const int d[3] = frag_i2d3(code);
    j = add_faces(j, d, /**/ fids);
    j = add_edges(j, d, /**/ fids);
    j = add_cornr(j, d, /**/ fids);
    return j;
}

__global__ void build_map(int3 L, int soluteid, int n, const Particle *pp, /**/ Map map) {
    int pid, fid, fids[MAX_DSTS], ndsts, j;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= n) return;
    const Particle p = pp[pid];

    fid = map_code(L, p.r);
    ndsts = map_decode(fid, /**/ fids);

    for (j = 0; j < ndsts; ++j)
        add_to_map(soluteid, pid, fids[j], /**/ map);
}

static __device__ void warpexscan(int cnt, int t, /**/ int *starts) {
    int L, scan;
    scan = cnt;
    for (L = 1; L < 32; L <<= 1) scan += (t >= L) * __shfl_up(scan, L);
    if (t < 27) starts[t] = scan - cnt;
}

__global__ void scan2d(const int *counts, const int *oldtcounts, /**/ int *tcounts, int *starts) {
    int t, cnt, newcnt;
    t = threadIdx.x;
    cnt = 0;
    if (t < 26) {
        cnt = counts[t];
        newcnt = cnt + oldtcounts[t];
        tcounts[t] = newcnt;
    }
    if (starts) warpexscan(cnt, t, /**/ starts);
}

__global__ void scan1d(const int *count, /**/ int *starts) {
    int t, cnt;
    t = threadIdx.x;
    cnt = 0;
    if (t < 26) cnt = count[t];
    if (starts) warpexscan(cnt, t, /**/ starts);
}


static __device__ void pack_p(const Particle *pp, int offset, int *indices, int i, /**/ Particle *buf) {
    int dst, src;
    dst = offset + i;
    src = __ldg(indices + dst);
    buf[dst] = pp[src];
}

__global__ void pack_pp(const Particle *pp, PackHelper ph, /**/ Pap26 buf) {
    int gid, fid, frag_i, hi, step;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    hi = ph.starts[26];
    step = gridDim.x * blockDim.x;

    for (  ; gid < hi; gid += step) {
        fid = k_common::fid(ph.starts, gid);

        /* index in the fragment coordinates */ 
        frag_i = gid - ph.starts[fid];
        
        pack_p(pp, ph.offsets[fid], ph.indices[fid], frag_i, /**/ buf.d[fid]);
    }
}

// TODO this is copy/paste from distr/common
static __device__ void fid2shift(int id, /**/ int s[3]) {
    enum {X, Y, Z};
    s[X] = XS * frag_i2d(id, X);
    s[Y] = YS * frag_i2d(id, Y);
    s[Z] = ZS * frag_i2d(id, Z);
}

static  __device__ void shift_1p(const int s[3], /**/ Particle *p) {
    enum {X, Y, Z};
    p->r[X] += s[X];
    p->r[Y] += s[Y];
    p->r[Z] += s[Z];
}

__global__ void shift_one_frag(int n, const int fid, /**/ Particle *pp) {
    int i, s[3];
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    
    fid2shift(fid, /**/ s);
    shift_1p(s, /**/ pp + i);
}


static __device__ void unpack_f(const Force *hff, int offset, int *indices, int i, /**/ Force *ff) {
    enum {X, Y, Z};
    int dst, src;

    src = offset + i;
    dst = __ldg(indices + src);

    const Force f = hff[src];
    
    atomicAdd(ff[dst].f + X, f.f[X]);
    atomicAdd(ff[dst].f + Y, f.f[Y]);
    atomicAdd(ff[dst].f + Z, f.f[Z]);
}

__global__ void unpack_ff(Fop26 hff, PackHelper ph, /**/ Force *ff) {
    int gid, fid, frag_i, hi, step;
    gid = threadIdx.x + blockDim.x * blockIdx.x;
    hi = ph.starts[26];
    step = gridDim.x * blockDim.x;

    for (  ; gid < hi; gid += step) {
        fid = k_common::fid(ph.starts, gid);

        /* index in the fragment coordinates */ 
        frag_i = gid - ph.starts[fid];
        
        unpack_f(hff.d[fid], ph.offsets[fid], ph.indices[fid], frag_i, /**/ ff);
    }
}

} // dev
