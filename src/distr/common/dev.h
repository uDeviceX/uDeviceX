namespace dev {

/* pack packets of nv particles into 27 buffers according to map  */
__global__ void pack_pp_packets(int nv, const Particle *pp, DMap m, /**/ Sarray<Particle*, 27> buf) {
    int i, cid, fid, scid;
    int dst, src, offset;
    i   = threadIdx.x + blockDim.x * blockIdx.x;
    cid = blockIdx.y;

    if (i >= nv) return;
    fid = frag_get_fid(m.starts, cid);

    offset = cid - m.starts[fid];
    scid   = m.ids[fid][offset];
    
    dst = nv * offset + i; 
    src = nv * scid   + i;
    
    buf.d[fid][dst] = pp[src];
}

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

__global__ void shift_halo(const Sarray<int, 27> starts, /**/ Particle *pp) {
    int pid, fid, s[3];

    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= starts.d[26]) return;
    fid = frag_get_fid(starts.d, pid);
    
    fid2shift(fid, s);
    shift_1p(s, /**/ pp + pid);
}

} // dev
