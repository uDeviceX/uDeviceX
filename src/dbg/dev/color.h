static __device__ err_type valid_color(int c, bool verbose) {
    if (c != BLUE_COLOR && c != RED_COLOR) {
        if (verbose) printf("DBG: color = %d\n", c);
        return ERR_INVALID;
    }
    return ERR_NONE;
}

__global__ void check_cc(const int *cc, int n, bool verbose = true) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    err_type e = valid_color(cc[i], verbose);
    report(e);
}
