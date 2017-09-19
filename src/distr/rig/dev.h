namespace dev {

__global__ void build_map(int n, const Solid *ss, /**/ Map m) {
    enum {X, Y, Z};
    int i, fid;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float r[3];

    r[X] = ss[i].com[X];
    r[Y] = ss[i].com[Y];
    r[Z] = ss[i].com[Z];
    
    fid = get_fid(r);
    add_to_map(i, fid, /**/ m);
}

__global__ void pack_pp(int nv, const Particle *pp, Map m, /**/ Sarray<Particle*, 27> buf) {
    int i, sid, fid, ssid;
    int dst, src, offset;
    i   = threadIdx.x + blockDim.x * blockIdx.x;
    sid = blockIdx.y;

    if (i >= nv) return;
    fid = k_common::fid(m.starts, sid);

    offset = sid - m.starts[fid];
    ssid    = m.ids[fid][offset];
    
    dst = nv * offset + i; 
    src = nv * ssid   + i;
    
    buf.d[fid][dst] = pp[src];
}

__global__ void pack_ss(int n, const Solid *ss, Map m, /**/ Sarray<Solid*, 27> buf) {
    int i, cid, fid;
    int dst, src, offset;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (i >= n) return;
    fid = k_common::fid(m.starts, i);
    
    dst = i - m.starts[fid];; 
    src = m.ids[fid][offset];
    
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

__global__ void shift(int n, const int fid, /**/ Particle *pp) {
    int i, s[3];
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    
    fid2shift(fid, /**/ s);
    shift_1p(s, /**/ pp + i);
}

} //dev
