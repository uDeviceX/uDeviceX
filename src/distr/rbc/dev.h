namespace dev {

__global__ void build_map(const float *rr, const int n, /**/ Map m) {
    int i, fid;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    const float *r = rr + 3 * i;

    fid = get_fid(r);
    add_to_map(i, fid, /**/ m);
}

} //dev
