static void get_num_capacity(/**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = MAX_SOLIDS;
}

void ini(int nv, Pack *p) {
    int numc[NBAGS];
    get_num_capacity(/**/ numc);

    ini_map(NBAGS, numc, /**/ &p->map);

    /* one datum is here a full mesh, so bsize is nv * sizeof(Particle) */
    ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ &p->hipp, &p->dipp);
    
    ini(PINNED, DEV_ONLY, sizeof(Solid), numc, /**/ &p->hss, &p->dss);
}

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c) {
    ini(comm, /*io*/ tg, /**/ &c->pp);
}

void ini(int nv, Unpack *u) {
    int numc[NBAGS];
    get_num_capacity(/**/ numc);

    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ &u->hpp, NULL);

    ini(HST_ONLY, NONE, sizeof(Solid), numc, /**/ &u->hss, NULL);
}
