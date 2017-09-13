static void scan(const int *counts, int n, /**/ int *starts) {
    scan::Work ws;
    scan::alloc_work(n, /**/ &ws);
    scan::scan(counts, n, /**/ starts, /*w*/ &ws);
    scan::free_work(&ws);
}

static void buildn(int n, int xcells, int ycells, int zcells,
                   /**/ Particle *pp, int *starts, int *counts) {
    if (!n) return;

    int ncells = xcells * ycells * zcells;
    if (!ncells) return;
    int3 cells = make_int3(xcells, ycells, zcells);

    int *ids;
    Particle *ppd;
    Dalloc(&ids, n);
    Dalloc(&ppd, n);

    CC(d::MemsetAsync(counts, 0, ncells * sizeof(int)));
    KL(dev::get_counts, (k_cnf(n)), (pp, n, cells, /**/ counts));
    scan(counts, ncells, /**/ starts);
    DzeroA(counts, ncells);
    KL(dev::get_ids, (k_cnf(n)), (pp, starts, n, cells, /**/ counts, ids));
    KL(dev::gather, (k_cnf(n)), (pp, ids, n, /**/ ppd));

    aD2D(pp, ppd, n);
    Dfree(ids);
    Dfree(ppd);
}

void ini(int X, int Y, int Z, /**/ Clist0 *c) {
    c->LX = X; c->LY = Y; c->LZ = Z;
    c->ncells = X * Y * Z + 1;
    Dalloc(&c->start, c->ncells);
    Dalloc(&c->count, c->ncells);
}

void fin(Clist0 *c) {
    Dfree(c->start); Dfree(c->count);
}

void build0(Clist0 *c, Particle *const pp, int n) {
    if (n)
        buildn(n, c->LX, c->LY, c->LZ, /**/ pp, c->start, c->count);
    else {
        DzeroA(c->start, c->ncells);
        DzeroA(c->count, c->ncells);
    }
}

Clist::Clist(int X, int Y, int Z) {
    LX = X; LY = Y; LZ = Z;
    ncells = LX * LY * LZ + 1;
    Dalloc(&start, ncells);
    Dalloc(&count, ncells);
}

void Clist::build(Particle *const pp, int n) {
    if (n)
        buildn(n, LX, LY, LZ, /**/ pp, start, count);
    else {
        DzeroA(start, ncells);
        DzeroA(count, ncells);
    }
}

Clist::~Clist() {
    Dfree(start); Dfree(count);
}
