// TODO this can be refined
static void get_capacity(int nfrags, int max_mesh_num, /**/ int cap[]) {
    for (int i = 0; i < nfrags; ++i)
        cap[i] = max_mesh_num;
}

void emesh_pack_ini(int nv, int max_mesh_num, EMeshPack *p) {
    int cap[NFRAGS];
    size_t msz = nv * sizeof(Particle);
    get_capacity(NFRAGS, max_mesh_num, /**/ cap);

    UC(emap_ini(1, NFRAGS, cap, /**/ &p->map));
    UC(bags_ini(PINNED, NONE, msz, cap, /**/ &p->hpp, &p->dpp));

    CC(d::Malloc((void**) &p->minext, max_mesh_num * sizeof(float3)));
    CC(d::Malloc((void**) &p->maxext, max_mesh_num * sizeof(float3)));
}

void emesh_comm_ini(MPI_Comm comm, /**/ EMeshComm *c) {
    UC(comm_ini(comm, /**/ &c->pp));
}

void emesh_unpack_ini(int nv, int max_mesh_num, EMeshUnpack *u) {
    int cap[NFRAGS];
    size_t msz = nv * sizeof(Particle);
    get_capacity(NFRAGS, max_mesh_num, /**/ cap);

    UC(bags_ini(PINNED_DEV, NONE, msz, cap, /**/ &u->hpp, &u->dpp));
}


/* Momentum structures */

static void get_mcap(int nfrags, int nt, const int *cap, /**/ int *mcap) {
    for (int i = 0; i < nfrags; ++i)
        mcap[i] = nt * cap[i];
}

static void ini_map(int nt, int max_mesh_num, MMap *map) {
    size_t sz = max_mesh_num * sizeof(int);
    CC(d::Malloc((void**) &map->cc, sz));
    CC(d::Malloc((void**) &map->ss, sz));
    CC(d::Malloc((void**) &map->subids, sz * nt));
}

void emesh_packm_ini(int nt, int max_mesh_num, EMeshPackM *p) {
    int i, cap[NFRAGS], mcap[NFRAGS];

    get_capacity(NFRAGS, max_mesh_num, /**/ cap);
    get_mcap(NFRAGS, nt, cap, /**/ mcap);

    for (i = 0; i < NFRAGS; ++i)
        UC(ini_map(nt, cap[i], /**/ &p->maps[i]));
    
    UC(bags_ini(PINNED,   NONE, sizeof(Momentum), mcap, /**/ &p->hmm, &p->dmm));
    UC(bags_ini(PINNED,   NONE, sizeof(int)     , mcap, /**/ &p->hii, &p->dii));

    CC(d::alloc_pinned((void**) &p->cchst, 26 * sizeof(int)));
    CC(d::HostGetDevicePointer((void**) &p->ccdev, p->cchst, 0));
}

void emesh_commm_ini(MPI_Comm comm, /**/ EMeshCommM *c) {
    UC(comm_ini(comm, /**/ &c->mm));
    UC(comm_ini(comm, /**/ &c->ii));
}

void emesh_unpackm_ini(int nt, int max_mesh_num, EMeshUnpackM *u) {
    int cap[NFRAGS], mcap[NFRAGS];

    get_capacity(NFRAGS, max_mesh_num, /**/ cap);
    get_mcap(NFRAGS, nt, cap, /**/ mcap);

    UC(bags_ini(PINNED_DEV, NONE, sizeof(Momentum), mcap, /**/ &u->hmm, &u->dmm));
    UC(bags_ini(PINNED_DEV, NONE, sizeof(int)     , mcap, /**/ &u->hii, &u->dii));
}
