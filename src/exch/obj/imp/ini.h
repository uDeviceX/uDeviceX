static void estimates(int nfrags, int maxd, int *cap) {
    frag_estimates(nfrags, maxd, /**/ cap);
}

void ini(int nw, int maxd, Pack *p) {
    int cap[NFRAGS];
    estimates(NFRAGS, maxd, /**/ cap);

    ini_map(nw, NFRAGS, cap, /**/ &p->map);
    ini(PINNED, NONE, sizeof(Particle), cap, /**/ &p->hpp, &p->dpp);
}

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c) {
    ini(comm, /*io*/ tg, /**/ &c->pp);
    ini(comm, /*io*/ tg, /**/ &c->ff);
}

void ini(int maxd, Unpack *u) {
    int cap[NFRAGS];
    frag_estimates(NFRAGS, maxd, /**/ cap);

    ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ &u->hpp, &u->dpp);
}

void ini(int maxd, PackF *p) {
    int cap[NFRAGS];
    estimates(NFRAGS, maxd, /**/ cap);

    ini(PINNED_DEV, NONE, sizeof(Force), cap, /**/ &p->hff, &p->dff);
}

void ini(int maxd, UnpackF *u) {
    int cap[NFRAGS];
    estimates(NFRAGS, maxd, /**/ cap);

    ini(PINNED_DEV, NONE, sizeof(Force), cap, /**/ &u->hff, &u->dff);
}
