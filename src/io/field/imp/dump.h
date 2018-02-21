static void create_dir(MPI_Comm comm) {
    if (m::is_master(comm))
        UC(os_mkdir(DUMP_BASE "/h5"));
    MC(m::Barrier(comm));
}

void io_field_ini(MPI_Comm comm, const Coords *c, IoField **iop) {
    enum {X, Y, Z};
    IoField *io;
    int nc;
    EMALLOC(1, iop);
    io = *iop;

    nc = xs(c) * ys(c) * zs(c);
    EMALLOC(nc, &io->rho);
    EMALLOC(nc, &io->u[X]);
    EMALLOC(nc, &io->u[Y]);
    EMALLOC(nc, &io->u[Z]);

    io->id = 0;
    create_dir(comm);
}

void io_field_fin(IoField *io) {
    enum {X, Y, Z};
    EFREE(io->rho);
    EFREE(io->u[X]);
    EFREE(io->u[Y]);
    EFREE(io->u[Z]);
    EFREE(io);
}

static int minmax(int lo, int hi, int x) {
    return \
        x <= lo ? lo :
        x >= hi ? hi :
        x;
}

static void zero0(float *a, int n) {
    int i;
    for (i = 0; i < n; i ++) a[i] = 0.0;
}

static void zero(float *rho, float *v[3], int n) {
    enum {X, Y, Z};
    zero0(rho, n);
    zero0(v[X], n); zero0(v[Y], n); zero0(v[Z], n);
}

static void avg(const Coords *coords, Particle *pp, int n, int nc, /**/
                float *rho, float *u[3]) {
    enum {X, Y, Z};
    int c, i, entry;
    float *r, *v;
    int Lx, Ly, Lz;

    Lx = xs(coords);
    Ly = ys(coords);
    Lz = zs(coords);

    zero(rho, u, nc);
    for (i = 0; i < n; ++i) {
        r = pp[i].r;
        v = pp[i].v;
        int index[3] = {
            minmax(0, Lx - 1, (int)(floor(r[X])) + Lx / 2),
            minmax(0, Ly - 1, (int)(floor(r[Y])) + Ly / 2),
            minmax(0, Lz - 1, (int)(floor(r[Z])) + Lz / 2)
        };
        entry = index[0] + Lx * (index[1] + Ly * index[2]);
        rho[entry] += 1;
        for (c = 0; c < 3; ++c) u[c][entry] += v[c];
    }
    for (c = 0; c < 3; ++c)
    for (i = 0; i < nc; ++i)
        u[c][i] = rho[i] ? u[c][i] / rho[i] : 0;
}

void io_field_dump_pp(const Coords *coords, MPI_Comm cart, IoField *io, int n, Particle *pp) {
    enum {X, Y, Z};
    char path[BUFSIZ];
    const char *names[] = { "density", "u", "v", "w" };
    int nc;
    float *data[] = { io->rho, io->u[X], io->u[Y], io->u[Z] };
    
    nc = xs(coords) * ys(coords) * zs(coords);
    avg(coords, pp, n, nc, io->rho, io->u);
    
    sprintf(path, DUMP_BASE "/h5/%04ld.h5", io->id++);
    
    UC(h5_write(coords, cart, path, data, names, 4));
    if (m::is_master(cart))
        xmf_write(coords, path, names, 4);
}
