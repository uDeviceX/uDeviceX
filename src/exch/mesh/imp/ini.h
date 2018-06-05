// TODO this can be refined
static void get_capacity(int nfrags, int max_mesh_num, /**/ int cap[]) {
    for (int i = 0; i < nfrags; ++i)
        cap[i] = max_mesh_num;
}

void emesh_pack_ini(bool prev_mesh, int3 L, int nv, int max_mesh_num, EMeshPack **pack) {
    int i, cap[NFRAGS];
    size_t msz = nv * sizeof(Particle);
    EMeshPack *p;

    EMALLOC(1, pack);
    p = *pack;

    p->L = L;
    get_capacity(NFRAGS, max_mesh_num, /**/ cap);

    UC(emap_ini(1, NFRAGS, cap, /**/ &p->map));

    p->hpp_prev = NULL; p->dpp_prev = NULL;
    i = 0;
    p->dpp = &p->dbags[i];
    p->hpp = &p->hbags[i++];    
    UC(comm_bags_ini(PINNED, NONE, msz, cap, /**/ p->hpp, p->dpp));

    if (prev_mesh) {
        p->dpp_prev = &p->dbags[i];
        p->hpp_prev = &p->hbags[i++];    
        UC(comm_bags_ini(PINNED, NONE, msz, cap, /**/ p->hpp_prev, p->dpp_prev));
    }

    p->nbags = i;

    Dalloc(&p->minext, max_mesh_num);
    Dalloc(&p->maxext, max_mesh_num);
}

void emesh_comm_ini(MPI_Comm cart, /**/ EMeshComm **com) {
    EMeshComm *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->pp));
}

void emesh_unpack_ini(bool prev_mesh, int3 L, int nv, int max_mesh_num, EMeshUnpack **unpack) {
    int i, cap[NFRAGS];
    size_t msz = nv * sizeof(Particle);
    EMeshUnpack *u;
    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    get_capacity(NFRAGS, max_mesh_num, /**/ cap);

    u->hpp_prev = NULL; u->dpp_prev = NULL;
    i = 0;
    u->hpp = &u->hbags[i];
    u->dpp = &u->dbags[i++];    
    UC(comm_bags_ini(PINNED_DEV, NONE, msz, cap, /**/ u->hpp, u->dpp));

    if (prev_mesh) {
        u->hpp_prev = &u->hbags[i];
        u->dpp_prev = &u->dbags[i++];    
        UC(comm_bags_ini(PINNED_DEV, NONE, msz, cap, /**/ u->hpp_prev, u->dpp_prev));
    }

    u->nbags = i;
}


/* Momentum structures */

static void get_mcap(int nfrags, int nt, const int *cap, /**/ int *mcap) {
    for (int i = 0; i < nfrags; ++i)
        mcap[i] = nt * cap[i];
}

static void ini_map(int nt, int max_mesh_num, MMap *map) {
    Dalloc(&map->cc, max_mesh_num);
    Dalloc(&map->ss, max_mesh_num);
    Dalloc(&map->subids, max_mesh_num * nt);
}

void emesh_packm_ini(int nt, int max_mesh_num, EMeshPackM **pack) {
    int i, cap[NFRAGS], mcap[NFRAGS];
    EMeshPackM *p;

    EMALLOC(1, pack);
    p = *pack;

    get_capacity(NFRAGS, max_mesh_num, /**/ cap);
    get_mcap(NFRAGS, nt, cap, /**/ mcap);

    for (i = 0; i < NFRAGS; ++i)
        UC(ini_map(nt, cap[i], /**/ &p->maps[i]));

    p->hmm = &p->hbags[ID_MM]; p->hii = &p->hbags[ID_II];
    p->dmm = &p->dbags[ID_MM]; p->dii = &p->dbags[ID_II];
    
    UC(comm_bags_ini(PINNED,   NONE, sizeof(Momentum), mcap, /**/ p->hmm, p->dmm));
    UC(comm_bags_ini(PINNED,   NONE, sizeof(int)     , mcap, /**/ p->hii, p->dii));

    p->nbags = MAX_NMBAGS;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));
    
    CC(d::alloc_pinned((void**) &p->cchst, 26 * sizeof(int)));
    CC(d::HostGetDevicePointer((void**) &p->ccdev, p->cchst, 0));
}

void emesh_commm_ini(MPI_Comm cart, /**/ EMeshCommM **com) {
    EMeshCommM *c;
    EMALLOC(1, com);
    c = *com;    
    UC(comm_ini(cart, /**/ &c->comm));
}

void emesh_unpackm_ini(int nt, int max_mesh_num, EMeshUnpackM **unpack) {
    int cap[NFRAGS], mcap[NFRAGS];
    EMeshUnpackM *u;
    EMALLOC(1, unpack);
    u = *unpack;

    get_capacity(NFRAGS, max_mesh_num, /**/ cap);
    get_mcap(NFRAGS, nt, cap, /**/ mcap);

    u->hmm = &u->hbags[ID_MM]; u->hii = &u->hbags[ID_II];
    u->dmm = &u->dbags[ID_MM]; u->dii = &u->dbags[ID_II];

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Momentum), mcap, /**/ u->hmm, u->dmm));
    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(int)     , mcap, /**/ u->hii, u->dii));

    u->nbags = MAX_NMBAGS;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
}
