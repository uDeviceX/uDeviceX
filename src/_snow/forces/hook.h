namespace forces {
static __device__ void hook(int ca, int cb, float *fx, float *fy, float *fz) {
    bool cond;
    cond = (ca == RED_COLOR || cb == RED_COLOR);
    *fx =  cond ? 1 : 0;
}

} /* namespace */
