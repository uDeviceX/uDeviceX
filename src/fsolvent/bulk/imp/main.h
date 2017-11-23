static void setup_flocal0() {
    setup_cloud();
    if (multi_solvent) setup_cloud_color();
    setup();
}

static void setup_flocal() {
    static bool fdpd_init = false;
    if (!fdpd_init) {
        setup_flocal0();
        fdpd_init = true;
    }
}

static void ini_flocal(const float4 *zip0, const ushort4 *zip1, int np, const int *start, const int *count, float seed, float *ff) {
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
        CC(d::PeekAtLastError());
    } else {
        fprintf(stderr, "Incompatible grid config\n");
    }
}

static void ini_flocal_color(const int *cc, int n) {
    ini_cloud_color(cc, n);
}

void flocal(const float4 *zip0, const ushort4 *zip1, int n, const int *start, const int *count,
            rnd::KISS* rnd, /**/ Force *ff) {
    if (n <= 0) return;
    setup_flocal();
    ini_flocal(zip0, zip1, n, start, count, rnd->get_float(), (float*)ff);
    launch(n);
    transpose(n, ff);
}

void flocal_color(const float4 *zip0, const ushort4 *zip1, const int *colors, int n, const int *start, const int *count,
                  rnd::KISS* rnd, /**/ Force *ff) {
    if (n <= 0) return;
    setup_flocal();
    ini_flocal(zip0, zip1, n, start, count, rnd->get_float(), (float*)ff);
    ini_flocal_color(colors, n);
    launch(n);
    transpose(n, ff);
}
