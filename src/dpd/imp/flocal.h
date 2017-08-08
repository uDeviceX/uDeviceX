void flocal0(float4 *zip0, ushort4 *zip1, int np, int *start, int *count, float seed, float* ff) {
    static InfoDPD c;
    if(!fdpd_init) {
        setup();
        fdpd_init = true;
    }
    tex(zip0, zip1, np, start, count);
    c.ncells = make_int3(XS, YS, ZS);
    c.nxyz = XS * YS * ZS;
    c.ff = ff;
    c.seed = seed;
    CC(cudaMemcpyToSymbol(info, &c, sizeof(c), 0, H2D));

    int np32 = np;
    if(np32 % 32) np32 += 32 - np32 % 32;
    CC(cudaMemsetAsync(ff, 0, sizeof(float)* np32 * 3));

    if(XS % MYCPBX == 0 && YS % MYCPBY == 0 && ZS % MYCPBZ == 0) {
        merged<<<dim3(XS / MYCPBX, YS / MYCPBY, ZS / MYCPBZ), dim3(32, MYWPB), 0>>>();
        CC(cudaPeekAtLastError());
        transpose<<< 28, 1024, 0>>>(np);
        CC(cudaPeekAtLastError());
    } else {
        fprintf(stderr, "Incompatible grid config\n");
    }
}

void flocal(float4 *zip0, ushort4 *zip1, int n, int *start, int *count,
	    rnd::KISS* rnd, /**/ Force *ff) {
    if (n <= 0) return;
    flocal0(zip0, zip1, n, start, count, rnd->get_float(), (float*)ff);
}
