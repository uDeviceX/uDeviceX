namespace forces {
static __device__ void hook(int ca, int cb, float x, float y, float z, float *fx, float *fy, float *fz) {
    bool cond;
    cond = (ca == RED_COLOR || cb == RED_COLOR);
    *fx =  cond ? -1e-6 : 1e-6;
    if (z > -1 && z < 1)
        printf("%d %d    %g %g %g :snow:\n", ca, cb, x, y, z);
}

} /* namespace */
