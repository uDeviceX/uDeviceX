namespace clist {

struct Clist {
    int3 dims;
    int ncells;
    int *starts, *counts;
};

struct Ticket {
    uchar4 *eelo, *eere; /* cell entries */
    uint *ii;
    scan::Work scan;
};

void ini(int LX, int LY, int LZ, /**/ Clist *c);
void fin(/**/ Clist *c);

void ini_ticket(const Clist *c, /**/ Ticket *t);
void fin_ticket(Ticket *t);

void build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, Ticket *t);
void build(int nlo, int nre, int nout, const Particle *pplo, const Particle *ppre, /**/ Particle *ppout, Clist *c, Ticket *t);

} /* namespace */
