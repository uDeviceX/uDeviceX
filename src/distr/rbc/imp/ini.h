static void get_num_capacity(int maxnc, /**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = maxnc;
}

void drbc_pack_ini(int maxnc, int nv, DRbcPack **pack) {
    int numc[NBAGS];
    DRbcPack *p;
    UC(emalloc(sizeof(DRbcPack), (void**) pack));
    p = *pack;
    
    get_num_capacity(maxnc, /**/ numc);

    UC(dmap_ini(NBAGS, numc, /**/ &p->map));

    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    UC(bags_ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ &p->hpp, &p->dpp));

    CC(d::Malloc((void**) &p->minext, maxnc * sizeof(float3)));
    CC(d::Malloc((void**) &p->maxext, maxnc * sizeof(float3)));

    if (rbc_ids) {
        UC(dmap_ini_host(NBAGS, numc, /**/ &p->hmap));
        UC(bags_ini(HST_ONLY, HST_ONLY, sizeof(int), numc, /**/ &p->hii, NULL));
    }
}

void drbc_comm_ini(MPI_Comm cart, /**/ DRbcComm **com) {
    DRbcComm *c;
    UC(emalloc(sizeof(DRbcComm), (void**) com));
    c = *com;
    
    UC(comm_ini(cart, /**/ &c->pp));
    if (rbc_ids)
        UC(comm_ini(cart, /**/ &c->ii));
}

void drbc_unpack_ini(int maxnc, int nv, DRbcUnpack **unpack) {
    int numc[NBAGS];
    DRbcUnpack *u;
    UC(emalloc(sizeof(DRbcUnpack), (void**) unpack));
    u = *unpack;

    get_num_capacity(maxnc, /**/ numc);

    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    UC(bags_ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ &u->hpp, NULL));

    if (rbc_ids)
        UC(bags_ini(HST_ONLY, NONE, sizeof(int), numc, /**/ &u->hii, NULL));
}
