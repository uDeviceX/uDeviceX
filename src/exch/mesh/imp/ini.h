// TODO this can be refined
static void get_capacity(int nfrags, int max_mesh_num, /**/ int cap[]) {
    for (int i = 0; i < nfrags; ++i)
        cap[i] = max_mesh_num;
}

void ini(int nv, int max_mesh_num, Pack *p) {
    int cap[NFRAGS];
    size_t msz = nv * sizeof(Particle);
    get_capacity(NFRAGS, max_mesh_num, /**/ cap);

    ini_map(1, NFRAGS, cap, /**/ &p->map);
    ini(PINNED, NONE, msz, cap, /**/ &p->hpp, &p->dpp);

    CC(d::Malloc((void**) &p->minext, max_mesh_num * sizeof(float3)));
    CC(d::Malloc((void**) &p->maxext, max_mesh_num * sizeof(float3)));
}

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c) {
    ini(comm, /*io*/ tg, /**/ &c->pp);
}

void ini(int nv, int max_mesh_num, Unpack *u) {
    int cap[NFRAGS];
    size_t msz = nv * sizeof(Particle);
    get_capacity(NFRAGS, max_mesh_num, /**/ cap);

    ini(PINNED_DEV, NONE, msz, cap, /**/ &u->hpp, &u->dpp);
}


/* Momentum structures */

static void get_mcap(int nfrags, int nt, const int *cap, /**/ int *mcap) {
    for (int i = 0; i < nfrags; ++i)
        mcap[i] = nt * cap[i];
}

void ini(int nt, int max_mesh_num, PackM *p) {
    int cap[NFRAGS], mcap[NFRAGS];

    get_capacity(NFRAGS, max_mesh_num, /**/ cap);
    get_mcap(NFRAGS, nt, cap, /**/ mcap);
    
    ini(PINNED,   NONE, sizeof(Momentum), mcap, /**/ &p->hmm, &p->dmm);
    ini(PINNED,   NONE, sizeof(int)     , mcap, /**/ &p->hii, &p->dii);
    ini(HST_ONLY, NONE, sizeof(int)     , cap , /**/ &p->hcc, NULL);
}

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ CommM *c) {
    ini(comm, /*io*/ tg, /**/ &c->mm);
    ini(comm, /*io*/ tg, /**/ &c->ii);
    ini(comm, /*io*/ tg, /**/ &c->cc);
}

void ini(int nt, int max_mesh_num, UnpackM *u) {
    int cap[NFRAGS], mcap[NFRAGS];

    get_capacity(NFRAGS, max_mesh_num, /**/ cap);
    get_mcap(NFRAGS, nt, cap, /**/ mcap);

    ini(PINNED_DEV, NONE, sizeof(Momentum), mcap, /**/ &u->hmm, &u->dmm);
    ini(PINNED_DEV, NONE, sizeof(int)     , mcap, /**/ &u->hii, &u->dii);
    ini(HST_ONLY,   NONE, sizeof(int)     , cap , /**/ &u->hcc, NULL);
}
