static void get_num_capacity(/**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = MAX_CELL_NUM;
}

static void get_capacity(int nv, const int numc[NBAGS], /**/ int capacity[NBAGS]) {
    for (int i = 0; i < NBAGS; ++i)
        capacity[i] = nv * numc[i];
}

void ini(int nv, Pack *p) {
    int numc[NBAGS], capacity[NBAGS];
    get_num_capacity(/**/ numc);
    get_capacity(nv, numc, /**/ capacity);

    ini_map(NBAGS, numc, /**/ &p->map);
    ini(PINNED, DEV_ONLY, sizeof(Particle), capacity, /**/ &p->hpp, &p->dpp);

    CC(d::Malloc((void**) &p->minext, MAX_CELL_NUM * sizeof(float3)));
    CC(d::Malloc((void**) &p->maxext, MAX_CELL_NUM * sizeof(float3)));
}

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c) {
    ini(comm, /*io*/ tg, /**/ &c->pp);
}

void ini(int nv, Unpack *u) {
    int numc[NBAGS], capacity[NBAGS];
    get_num_capacity(/**/ numc);
    get_capacity(nv, numc, /**/ capacity);

    ini(HST_ONLY, NONE, sizeof(Particle), capacity, /**/ &u->hpp, NULL);
}
