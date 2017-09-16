namespace dev {

__global__ void build_map(int n, const float3 *minext, const float3 *maxext, /**/ Map m) {
    enum {X, Y, Z};
    int i, fid;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float r[3];
    float3 mine, maxe;

    mine = minext[i];
    maxe = maxext[i];

    r[X] = 0.5f * (mine.x + maxe.x);
    r[Y] = 0.5f * (mine.y + maxe.y);
    r[Z] = 0.5f * (mine.z + maxe.z);
    
    fid = get_fid(r);
    add_to_map(i, fid, /**/ m);
}

__global__ void pack_pp(int nv, const Particle *pp, Map m, /**/ Sarray<Particle*, 27> buf) {
    int i, cid, fid;
    int dst, src, offset;
    i   = threadIdx.x + blockDim.x * blockIdx.x;
    cid = blockIdx.y;

    if (i >= nv) return;
    fid = k_common::fid(m.starts, cid);

    offset = cid - m.starts[fid];
    
    dst = nv * offset + i; 
    src = nv * cid    + i;
    
    buf.d[fid][dst] = pp[src];
}

} //dev
