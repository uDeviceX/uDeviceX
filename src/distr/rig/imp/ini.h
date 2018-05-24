static void get_num_capacity(int maxns, /**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = maxns;
}

void drig_pack_ini(int3 L, int maxns, int nv, DRigPack **pack) {
    int numc[NBAGS];
    DRigPack *p;
    EMALLOC(1, pack);
    p = *pack;

    p->L = L;
    get_num_capacity(maxns, /**/ numc);

    UC(dmap_ini(NBAGS, numc, /**/ &p->map));

    p->hipp = &p->hbags[ID_PP];  p->hss = &p->hbags[ID_SS];
    p->dipp = &p->dbags[ID_PP];  p->dss = &p->dbags[ID_SS];
    
    /* one datum is here a full mesh, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ p->hipp, p->dipp));
    UC(comm_bags_ini(PINNED, DEV_ONLY, sizeof(Solid), numc, /**/ p->hss, p->dss));

    p->nbags = MAX_NBAGS;

    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));
}

void drig_comm_ini(MPI_Comm cart, /**/ DRigComm **com) {
    DRigComm *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

void drig_unpack_ini(int3 L, int maxns, int nv, DRigUnpack **unpack) {
    int numc[NBAGS];
    DRigUnpack *u;
    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    get_num_capacity(maxns, /**/ numc);

    u->hipp = &u->hbags[ID_PP];  u->hss = &u->hbags[ID_SS];
    
    /* one datum is here a full mesh, so bsize is nv * sizeof(Particle) */
    UC(comm_bags_ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ u->hipp, NULL));

    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Solid), numc, /**/ u->hss, NULL));

    u->nbags = MAX_NBAGS;

    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
}
