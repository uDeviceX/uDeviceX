static void get_num_capacity(int maxnc, /**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = maxnc;
}

void drbc_pack_ini(bool ids, int3 L, int maxnc, int nv, DRbcPack **pack) {
    int i, numc[NBAGS];
    DRbcPack *p;
    EMALLOC(1, pack);
    p = *pack;

    p->L= L;
    
    get_num_capacity(maxnc, /**/ numc);

    UC(dmap_ini(NBAGS, numc, /**/ &p->map));

    i = 0;
    p->hpp = &p->hbags[i];
    p->dpp = &p->dbags[i++];
    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ p->hpp, p->dpp));

    Dalloc(&p->minext, maxnc);
    Dalloc(&p->maxext, maxnc);

    p->hii = NULL;
    if (ids) {
        p->hii = &p->hbags[i++];
        UC(dmap_ini_host(NBAGS, numc, /**/ &p->hmap));
        UC(comm_bags_ini(HST_ONLY, HST_ONLY, sizeof(int), numc, /**/ p->hii, NULL));
    }
    p->nbags = i;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));
}

void drbc_comm_ini(bool ids, MPI_Comm cart, /**/ DRbcComm **com) {
    DRbcComm *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

void drbc_unpack_ini(bool ids, int3 L, int maxnc, int nv, DRbcUnpack **unpack) {
    int i, numc[NBAGS];
    DRbcUnpack *u;
    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    
    get_num_capacity(maxnc, /**/ numc);

    i = 0;
    u->hpp = &u->hbags[i++];
    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ u->hpp, NULL));

    u->hii = NULL;
    if (ids) {
        u->hii = &u->hbags[i++];
        UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), numc, /**/ u->hii, NULL));
    }
    u->nbags = i;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
}
