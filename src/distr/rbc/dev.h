namespace dev {

__global__ void build_map(int n, const float3 *minext, const float3 *maxext, /**/ DMap m) {
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
    
    fid = dmap_get_fid(r);
    dmap_add(i, fid, /**/ m);
}

} //dev
