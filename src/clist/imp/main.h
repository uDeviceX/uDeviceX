static void scan(const int *counts, int n, /**/ int *starts) {
    scan::Work ws;
    scan::alloc_work(n, /**/ &ws);
    scan::scan(counts, n, /**/ starts, /*w*/ &ws);
    scan::free_work(&ws);
}

static void build(int n, int xcells, int ycells, int zcells,
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

void Clist::buildn(Particle *const pp, const int n) {
    clist::build(n, LX, LY, LZ, /**/ pp, start, count);
}

static void build0(int n, /**/ int *start, int *count) {
    DzeroA(start, n);
    DzeroA(count, n);
}

Clist::Clist(int X, int Y, int Z) {
    LX = X; LY = Y; LZ = Z;
    ncells = LX * LY * LZ + 1;
    Dalloc(&start, ncells);
    Dalloc(&count, ncells);
}

void Clist::build(Particle *const pp, int n) {
    if (n)
        buildn(pp, n);
    else
        build0(ncells, /**/ start, count);
}

Clist::~Clist() {
    Dfree(start); Dfree(count);
}
