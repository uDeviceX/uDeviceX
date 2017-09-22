namespace dev {

__global__ void build_map(int3 L, int soluteid, int n, const float3 *minext, const float3 *maxext, /**/ Map map) {
    int i, fid, fids[MAX_DSTS], ndsts, j;
    float3 lo, hi;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    lo = minext[i];
    hi = maxext[i];
    
    fid = map_code_box(L, lo, hi);
    ndsts = map_decode(fid, /**/ fids);

    for (j = 0; j < ndsts; ++j)
        add_to_map(soluteid, i, fids[j], /**/ map);
}



} // dev
