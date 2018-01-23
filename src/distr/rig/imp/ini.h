static void get_num_capacity(int maxns, /**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = maxns;
}

void drig_pack_ini(int maxns, int nv, DRigPack **pack) {
    int numc[NBAGS];
    DRigPack *p;
    UC(emalloc(sizeof(DRigPack), (void**) pack));
    p = *pack;
    get_num_capacity(maxns, /**/ numc);

    UC(dmap_ini(NBAGS, numc, /**/ &p->map));

    /* one datum is here a full mesh, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ &p->hipp, &p->dipp));
    
    UC(comm_bags_ini(PINNED, DEV_ONLY, sizeof(Solid), numc, /**/ &p->hss, &p->dss));
}

void drig_comm_ini(MPI_Comm cart, /**/ DRigComm **com) {
    DRigComm *c;
    UC(emalloc(sizeof(DRigComm), (void**) com));
    c = *com;
    UC(comm_ini(cart, /**/ &c->ipp));
    UC(comm_ini(cart, /**/ &c->ss));
}

void drig_unpack_ini(int maxns, int nv, DRigUnpack **unpack) {
    int numc[NBAGS];
    DRigUnpack *u;
    UC(emalloc(sizeof(DRigUnpack), (void**) unpack));
    u = *unpack;
    get_num_capacity(maxns, /**/ numc);

    /* one datum is here a full mesh, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ &u->hipp, NULL));

    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Solid), numc, /**/ &u->hss, NULL));
}
