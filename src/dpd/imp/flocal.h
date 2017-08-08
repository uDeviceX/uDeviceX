void flocal0(float4 *zip0, ushort4 *zip1, int np, int *start, int *count, float seed, float *ff) {
    if(!fdpd_init) {
        setup();
        fdpd_init = true;
    }
    tex(zip0, zip1, np, start, count);
    set_info(ff, np, seed);

    if (XS % MYCPBX == 0 && YS % MYCPBY == 0 && ZS % MYCPBZ == 0) {
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
