struct TClist {
    int *ss, *cc, *ii; /* starts, counts, ids */
};

void ini(int max_num_mesh, /**/ TClist *tc);
void fin(/**/ TClist *tc);

void build(int nt, int nv, const int4 *tt, const int nm, const Particle *pp, /**/ TClist *tc, /*w*/ scan::Work *w);
