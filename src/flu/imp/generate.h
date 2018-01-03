static void assert_np(int n, int m) {
    if (n > m) ERR("too many particles: n = %d < m = %d", n, m);
}

static int gen0(Particle *pp) { /* generate particle positions and velocities */
    enum {X, Y, Z};
    UC(assert_np(XS * YS * ZS * numberdensity, MAX_PART_NUM));
    os::srand(123456);
    int iz, iy, ix, l, nd = numberdensity;
    int n = 0; /* particle index */
    float x, y, z, dr = 0.99;
    for (iz = 0; iz < ZS; iz++)
    for (iy = 0; iy < YS; iy++)
    for (ix = 0; ix < XS; ix++) {
        /* edge of a cell */
        int xlo = -0.5*XS + ix, ylo = -0.5*YS + iy, zlo = -0.5*ZS + iz;
        for (l = 0; l < nd; l++) {
            Particle p;
            x = xlo + dr * os::drand(), y = ylo + dr * os::drand(), z = zlo + dr * os::drand();
            p.r[X] = x; p.r[Y] = y; p.r[Z] = z;
            p.v[X] = 0; p.v[Y] = 0; p.v[Z] = 0;
            pp[n++] = p;
        }
    }
    msg_print("ic::gen: %d solvent particles", n);
    return n;
}

static int genColor(Coords coords, /*o*/ Particle *pp, int *color, /*w*/ Particle *pp_hst, int *color_hst) {
    int n = gen0(pp_hst);
    inter::color_hst(coords, pp_hst, n, /**/ color_hst);
    cH2D(color, color_hst, n);
    cH2D(   pp,    pp_hst, n);
    return n;
}

static int genGrey(/*o*/ Particle *dev, /*w*/ Particle *hst) {
    int n = gen0(hst);
    cH2D(dev, hst, n);
    return n;
}

void gen_quants(Coords coords, Quants *q) {
    if (multi_solvent)
        q->n = genColor(coords, q->pp, q->cc, /*w*/ q->pp_hst, q->cc_hst);
    else
        q->n = genGrey(q->pp, /*w*/ q->pp_hst);
}

static void ii_gen0(MPI_Comm comm, const long n, int *ii) {
    long i0 = 0;
    MC(m::Exscan(&n, &i0, 1, MPI_LONG, MPI_SUM, comm));
    for (long i = 0; i < n; ++i) ii[i] = i + i0;
}

static void ii_gen(MPI_Comm comm, const int n, int *ii_dev, int *ii_hst) {
    ii_gen0(comm, n, ii_hst);
    cH2D(ii_dev, ii_hst, n);
}

void gen_ids(MPI_Comm comm, const int n, Quants *q) {
    ii_gen(comm, n, q->ii, q->ii_hst);
}
