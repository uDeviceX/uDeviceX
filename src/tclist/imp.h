namespace tclist {

struct TClist {
    int3 dims;
    int ncells;
    int *ss, *cc, *ii; /* starts, counts, ids */
};

void ini(int Lx, int Ly, int Lz, int maxtriangles, /**/ TClist *c);
void fin(/**/ TClist *c);

void reini(/**/ TClist *c);
void add_triangles(int nt, int nv, const int4 *tt, const int nm, const Particle *pp, /**/ TClist *c);
void scan(/**/ TClist *c, /*w*/ scan::Work *w);
void fill(int nt, int nv, const int4 *tt, const int nm, const Particle *pp, /**/ TClist *c);

} // tclist
