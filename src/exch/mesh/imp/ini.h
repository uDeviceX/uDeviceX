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
