static void get_num_capacity(/**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = MAX_CELL_NUM;
}

void ini(int nv, Pack *p) {
    int numc[NBAGS];
    get_num_capacity(/**/ numc);

    ini_map(NBAGS, numc, /**/ &p->map);

    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ &p->hpp, &p->dpp);

    CC(d::Malloc((void**) &p->minext, MAX_CELL_NUM * sizeof(float3)));
    CC(d::Malloc((void**) &p->maxext, MAX_CELL_NUM * sizeof(float3)));

    if (rbc_ids) {
        ini_host_map(NBAGS, numc, /**/ &p->hmap);
        ini(HST_ONLY, HST_ONLY, sizeof(int), numc, /**/ &p->hii, NULL);
    }
}

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c) {
    ini(comm, /*io*/ tg, /**/ &c->pp);
    if (rbc_ids)
        ini(comm, /*io*/ tg, /**/ &c->ii);
}

void ini(int nv, Unpack *u) {
    int numc[NBAGS];
    get_num_capacity(/**/ numc);

    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ &u->hpp, NULL);

    if (rbc_ids)
        ini(HST_ONLY, NONE, sizeof(int), numc, /**/ &u->hii, NULL);
}
