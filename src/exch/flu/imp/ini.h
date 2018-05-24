void eflu_pack_ini(bool colors, int3 L, int maxd, EFluPack **pack) {
    int i, nbags, nc, cap[NBAGS], ncs[NBAGS];
    EFluPack *p;

    EMALLOC(1, pack);
    p = *pack;

    p->L = L;
    frag_hst::estimates(L, NFRAGS, maxd, /**/ cap);
    cap[BULK] = 0;
    memcpy(p->cap.d, cap, NFRAGS * sizeof(int));
    
    for (i = 0; i < NFRAGS; ++i) {
        ncs[i] = nc = frag_hst::ncell(L, i) + 1;
        Dalloc(&p->bcc.d[i], nc);
        Dalloc(&p->bss.d[i], nc);
        Dalloc(&p->fss.d[i], nc);
        
        Dalloc(&p->bii.d[i], cap[i]);
    }
    ncs[BULK] = 0;

    nbags = 0;
    p->hpp = &p->hbags[nbags];
    p->dpp = &p->dbags[nbags++];
    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ p->hpp, p->dpp));

    if (colors) {
        p->hcc = &p->hbags[nbags];
        p->dcc = &p->dbags[nbags++];
        UC(comm_bags_ini(PINNED_DEV, NONE,  sizeof(int), cap, /**/ p->hcc, p->dcc));
    }

    p->hfss = &p->hbags[nbags++];
    UC(comm_bags_ini(PINNED_HST, NONE, sizeof(int), ncs, /**/ p->hfss, NULL));
    p->nbags = nbags;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));
    
    memcpy(p->hfss->counts, ncs, sizeof(ncs));
    
    Dalloc(&p->counts_dev, NFRAGS);

    p->opt.colors = colors;
}

void eflu_comm_ini(MPI_Comm cart, /**/ EFluComm **com) {
    EFluComm *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

void eflu_unpack_ini(bool colors, int3 L, int maxd, EFluUnpack **unpack) {
    int nbags, i, cap[NBAGS], ncs[NBAGS];
    EFluUnpack *u;

    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    frag_hst::estimates(L, NFRAGS, maxd, /**/ cap);
    cap[BULK] = 0;
    
    for (i = 0; i < NFRAGS; ++i)
        ncs[i] = frag_hst::ncell(L, i) + 1;
    ncs[BULK] = 0;

    nbags = 0;
    
    u->hpp = &u->hbags[nbags];
    u->dpp = &u->dbags[nbags++];    
    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ u->hpp, u->dpp));

    if (colors) {
        u->hcc = &u->hbags[nbags];
        u->dcc = &u->dbags[nbags++];        
        UC(comm_bags_ini(PINNED_DEV, NONE,  sizeof(int), cap, /**/ u->hcc, u->dcc));
    }

    u->hfss = &u->hbags[nbags];
    u->dfss = &u->dbags[nbags++];        
    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(int), ncs, /**/ u->hfss, u->dfss));

    u->nbags = nbags;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
    
    u->opt.colors = colors;
}

