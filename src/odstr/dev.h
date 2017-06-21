/* which neighboring subdomain `p' belongs to? */
__device__ int box(Particle *p) {
    enum {X, Y, Z};
    int c;
    int vc[3]; /* vcode */
    float *r = p->r;
    int   L[3] = {XS, YS, ZS};
    for (c = 0; c < 3; ++c) vc[c] = (2 + (r[c] >= -L[c]/2) + (r[c] >= L[c]/2)) % 3;
    return vc[X] + 3 * (vc[Y] + 3 * vc[Z]);
}

__global__ void halo(Particle *pp, int n, /**/ int *iidx[], int size[]) {
    int pid, code, entry;
    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    Particle *p = &pp[pid];
    code = box(p);
    if (code > 0) {
        entry = atomicAdd(size + code, 1);
        iidx[code][entry] = pid;
    }
}

__global__ void scan(int n, int size[], /**/ int strt[], int size_pin[]) {
    int tid = threadIdx.x;
    int val = 0, cnt = 0;

    if (tid < 27) {
        val = cnt = size[threadIdx.x];
        if (tid > 0) size_pin[tid] = cnt;
    }

    for (int L = 1; L < 32; L <<= 1) val += (tid >= L) * __shfl_up(val, L) ;
    if (tid < 28) strt[tid] = val - cnt;
    if (tid == 26) {
        strt[tid + 1] = val;
        int nbulk = n - val;
        size_pin[0] = nbulk;
    }
}

__device__ int code(int a[], uint i) { /* where is `i' in sorted a[27]? */
    uint k1, k9, k3;
    k9 = 9 * (i >= a[          9]) + 9 * (i >= a[         18]);
    k3 = 3 * (i >= a[k9      + 3]) + 3 * (i >= a[k9      + 6]);
    k1 =     (i >= a[k9 + k3 + 1]) +     (i >= a[k9 + k3 + 2]);
    return k9 + k3 + k1;
}

__global__ void pack(float2 *pp, int *iidx[], int send_strt[], /**/ float2 *send_dev[]) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    int slot = gid / 3;

    int tid = threadIdx.x;

    __shared__ int start[28];

    if (tid < 28) start[tid] = send_strt[tid];
    __syncthreads();
    int idpack = code(start, slot);

    if (slot >= start[27]) return;

    int offset = slot - start[idpack];
    int pid = __ldg(iidx[idpack] + offset);

    int c = gid % 3;
    int d = c + 3 * offset;
    send_dev[idpack][d] = pp[c + 3 * pid];
}

__global__ void unpack(uint n_pa, float2 *recv[], int strt[], int strt_pa[],
                       /*io*/ int *counts,
                       /*o*/ float2 *pp, uchar4 *subi) {
    /* n_pa: `n' padded; strt_pa: start padded */
    uint warpid, laneid;
    uint lb, ub, db; /* local/unpack/distribute base */

    int c; /* code */
    float2 d0, d1, d2; /* data */
    uint nu; /* "n unpack" */
    int xi, yi, zi, cid, subindex;

    warpid = threadIdx.x / 32;
    laneid = threadIdx.x % 32;

    lb = 32 * (warpid + 4 * blockIdx.x);
    if (lb >= n_pa)  return;
    c = code(strt_pa, lb); /* find `lb' in strt_pa */
    ub = lb - strt_pa[c];

    nu = min(32, strt[c + 1] - strt[c] - ub);
    if (nu == 0) return;

    k_common::read_AOS6f(recv[c] + 3 * ub, nu, d0, d1, d2);

    if (laneid < nu) {
        d0.x += XS * ((c     + 1) % 3 - 1);
        d0.y += YS * ((c / 3 + 1) % 3 - 1);
        d1.x += ZS * ((c / 9 + 1) % 3 - 1);

        xi = (int)floor((double)d0.x + XS / 2);
        yi = (int)floor((double)d0.y + YS / 2);
        zi = (int)floor((double)d1.x + ZS / 2);

        cid = xi + XS * (yi + YS * zi);
        subindex = atomicAdd(counts + cid, 1);
    }

    db = strt[c] + ub;
    k_common::write_AOS6f(pp + 3 * db, nu, d0, d1, d2);
    if (laneid < nu) subi[db + laneid] = make_uchar4(xi, yi, zi, subindex);
}

__global__ void scatter(bool remote, uchar4 *subi, int n, int *start,
                        /**/ uint *iidx) {
    uint pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    uchar4 entry = subi[pid];
    int subindex = entry.w;

    if (subindex != 255) {
        int cid = entry.x + XS * (entry.y + YS * entry.z);
        int base = __ldg(start + cid);

        pid |= remote << 31;
        iidx[base + subindex] = pid;
    }
}

__forceinline__ __device__ void xchg_aos2f(int srclane0, int srclane1, int start, float& s0, float& s1) {
    float t0 = __shfl(s0, srclane0);
    float t1 = __shfl(s1, srclane1);

    s0 = start == 0 ? t0 : t1;
    s1 = start == 0 ? t1 : t0;
    s1 = __shfl_xor(s1, 1);
}

__forceinline__ __device__ void xchg_aos4f(int srclane0, int srclane1, int start, float3& s0, float3& s1) {
    xchg_aos2f(srclane0, srclane1, start, s0.x, s1.x);
    xchg_aos2f(srclane0, srclane1, start, s0.y, s1.y);
    xchg_aos2f(srclane0, srclane1, start, s0.z, s1.z);
}

__global__ void gather(float2  *pp_lo, float2  *pp_re, int n, uint *iidx,
                       /**/ float2  *pp, float4  *zip0, ushort4 *zip1) {
    /* pp_lo, pp_re, pp: local, remote and output particles */
    int warpid, tid, base, pid;
    bool valid, remote;
    uint spid;
    float2 d0, d1, d2; /* data */
    int nsrc, src0, src1, start, destbase;
    float3 s0, s1;

    warpid = threadIdx.x >> 5;
    tid = threadIdx.x & 0x1f;

    base = 32 * (warpid + 4 * blockIdx.x);
    pid = base + tid;

    valid = (pid < n);

    if (valid) spid = iidx[pid];

    if (valid) {
        remote = (spid >> 31) & 1;
        spid &= ~(1 << 31);
        if (remote) {
            d0 = __ldg(pp_re + 0 + 3 * spid);
            d1 = __ldg(pp_re + 1 + 3 * spid);
            d2 = __ldg(pp_re + 2 + 3 * spid);
        } else {
            d0 = pp_lo[0 + 3 * spid];
            d1 = pp_lo[1 + 3 * spid];
            d2 = pp_lo[2 + 3 * spid];
        }
    }
    nsrc = min(32, n - base);

    src0 = (32 * ((tid    ) & 0x1) + tid) >> 1;
    src1 = (32 * ((tid + 1) & 0x1) + tid) >> 1;
    start = tid % 2;
    destbase = 2 * base;

    s0 = make_float3(d0.x, d0.y, d1.x);
    s1 = make_float3(d1.y, d2.x, d2.y);

    xchg_aos4f(src0, src1, start, s0, s1);

    if (tid < 2 * nsrc)
    zip0[destbase + tid] = make_float4(s0.x, s0.y, s0.z, 0);

    if (tid + 32 < 2 * nsrc)
    zip0[destbase + tid + 32] = make_float4(s1.x, s1.y, s1.z, 0);

    if (tid < nsrc)
    zip1[base + tid] = make_ushort4(__float2half_rn(d0.x),
                                    __float2half_rn(d0.y),
                                    __float2half_rn(d1.x),
                                    0);
    k_common::write_AOS6f(pp + 3 * base, nsrc, d0, d1, d2);
}
