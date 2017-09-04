static void setup_flocal() {
    if (!fdpd_init) {
        setup_cloud();
        if (multi_solvent) setup_cloud_color();
        setup();
        fdpd_init = true;
    }
}

static void ini_flocal(float4 *zip0, ushort4 *zip1, int np, int *start, int *count, float seed, float *ff) {
    tex_cells(start, count);
    ini_cloud(zip0, zip1, np);
    set_info(ff, np, seed);
}

static void launch(int np) {
    int nx, ny, nz;
    if (XS % MYCPBX == 0 && YS % MYCPBY == 0 && ZS % MYCPBZ == 0) {
        nx = XS / MYCPBX;
        ny = YS / MYCPBY;
        nz = ZS / MYCPBZ;
        KL(merged, (dim3(nx, ny, nz), dim3(32, MYWPB)), ());
        KL(transpose, (28, 1024), (np));
        CC(cudaPeekAtLastError());
    } else {
        fprintf(stderr, "Incompatible grid config\n");
    }
}

static void ini_flocal_color(int *cc, int n) {
    ini_cloud_color(cc, n);
}

void flocal(float4 *zip0, ushort4 *zip1, int n, int *start, int *count,
	    rnd::KISS* rnd, /**/ Force *ff) {
    if (n <= 0) return;
    setup_flocal();
    ini_flocal(zip0, zip1, n, start, count, rnd->get_float(), (float*)ff);
    launch(n);
}

void flocal_color(float4 *zip0, ushort4 *zip1, int *colors, int n, int *start, int *count,
                  rnd::KISS* rnd, /**/ Force *ff) {
    if (n <= 0) return;
    setup_flocal();
    ini_flocal(zip0, zip1, n, start, count, rnd->get_float(), (float*)ff);
    ini_flocal_color(colors, n);
    launch(n);
}
