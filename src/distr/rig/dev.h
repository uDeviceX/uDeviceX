namespace dev {

__global__ void build_map(int3 L, int n, const Solid *ss, /**/ DMap m) {
    enum {X, Y, Z};
    int i, fid;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float r[3];

    r[X] = ss[i].com[X];
    r[Y] = ss[i].com[Y];
    r[Z] = ss[i].com[Z];
    
    fid = dmap_get_fid(L, r);
    dmap_add(i, fid, /**/ m);
}

__global__ void pack_ss(const Solid *ss, DMap m, /**/ Sarray<Solid*, 27> buf) {
    int i, fid; /* [f]ragment [id] */
    int d, s;
    
    i = threadIdx.x + blockDim.x * blockIdx.x;
    fid = frag_get_fid(m.starts, i);
    if (i >= m.starts[27]) return;

    d = i - m.starts[fid];
    s = __ldg(m.ids[fid] + d);
    
    buf.d[fid][d] = ss[s];
}

} //dev
