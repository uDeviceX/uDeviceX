static void get_num_capacity(int maxnc, /**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = maxnc;
}

void drbc_pack_ini(bool ids, int3 L, int maxnc, int nv, DRbcPack **pack) {
    int numc[NBAGS];
    DRbcPack *p;
    EMALLOC(1, pack);
    p = *pack;

    p->L= L;
    
    get_num_capacity(maxnc, /**/ numc);

    UC(dmap_ini(NBAGS, numc, /**/ &p->map));

    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ &p->hpp, &p->dpp));

    CC(d::Malloc((void**) &p->minext, maxnc * sizeof(float3)));
    CC(d::Malloc((void**) &p->maxext, maxnc * sizeof(float3)));

    if (ids) {
        UC(dmap_ini_host(NBAGS, numc, /**/ &p->hmap));
        UC(comm_bags_ini(HST_ONLY, HST_ONLY, sizeof(int), numc, /**/ &p->hii, NULL));
    }
    p->ids = ids;
}

void drbc_comm_ini(bool ids, MPI_Comm cart, /**/ DRbcComm **com) {
    DRbcComm *c;
    EMALLOC(1, com);
    c = *com;
    
    UC(comm_ini(cart, /**/ &c->pp));
    if (ids)
        UC(comm_ini(cart, /**/ &c->ii));
    c->ids = ids;
}

void drbc_unpack_ini(bool ids, int3 L, int maxnc, int nv, DRbcUnpack **unpack) {
    int numc[NBAGS];
    DRbcUnpack *u;
    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    u->ids = ids;
    
    get_num_capacity(maxnc, /**/ numc);

    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ &u->hpp, NULL));

    if (ids)
        UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), numc, /**/ &u->hii, NULL));
}
