namespace k_rex {
enum {FD = 3}; /* number of dimenshion in force */

__device__ void unpack0(int fid, int pif, int dim, /**/ float *ff) {
    /* fid: fragment id, pif: particle id in fragment coordinates */
    int entry, dpid;
    float f; /* force */
    entry = g::offsets[fid] + pif;
    f = __ldg(g::recvbags[fid] + dim + FD * entry);
    dpid = __ldg(g::indexes[fid] + entry);
    atomicAdd(ff + FD * dpid + dim, f);
}

__device__ void unpack1(int pid, int dim, /**/ float *ff) {
    int fid; /* fragment id */
    int pif; /* particle id in fragment coordinates */
    fid = k_common::fid(g::starts, pid);
    pif = pid - g::starts[fid];
    if (pif >= g::counts[fid]) return;
    unpack0(fid, pif, dim, /**/ ff);
}

__global__ void unpack(/**/ float *ff) {
    int n;
    int gid; /* global id */
    int pid; /* particle id */
    int dim; /* dimenshion of the force (x, y, z) */
    int lo, hi, step;
    n = g::starts[26];
    lo = threadIdx.x + blockDim.x * blockIdx.x;
    step = blockDim.x * gridDim.x;
    hi = FD*n;
    for (gid = lo; gid < hi; gid += step) {
        pid = gid / FD;
        if (pid >= n) return;
        dim = gid % FD;
        unpack1(pid, dim, /**/ ff);
    }
}
}
