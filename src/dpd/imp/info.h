static void set_info(float *ff, int np, float seed) {
    static InfoDPD c;
    c.ncells = make_int3(XS, YS, ZS);
    c.nxyz = XS * YS * ZS;
    c.ff = ff;
    c.seed = seed;
    CC(cudaMemcpyToSymbol(info, &c, sizeof(c), 0, H2D));
    int np32 = np;
    if(np32 % 32) np32 += 32 - np32 % 32;
    CC(cudaMemsetAsync(ff, 0, sizeof(float)* np32 * 3));
}
