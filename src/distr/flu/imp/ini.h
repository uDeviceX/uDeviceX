static int estimate_max_flux(int3 L, int fid, int maxd) {
    int e, nfaces, d[3];
    frag_hst::i2d3(fid, d);
    nfaces = abs(d[0]) + abs(d[1]) + abs(d[2]);
    e = maxd * frag_hst::ncell(L, fid) * nfaces;
    return e;
}

static void get_capacity(int3 L, int maxd, /**/ int capacity[NBAGS]) {
    for (int i = 0; i < NFRAGS; ++i)
        capacity[i] = estimate_max_flux(L, i, maxd);
    capacity[frag_bulk] = 0;
}

void dflu_pack_ini(bool colors, bool ids, int3 L, int maxdensity, DFluPack **pack) {
    DFluPack *p;
    int i, capacity[NBAGS];

    EMALLOC(1, pack);
    p = *pack;

    p->L = L;
    
    get_capacity(L, maxdensity, /**/ capacity);

    UC(dmap_ini(NFRAGS, capacity, /**/ &p->map));

    p->hii = p->hcc = NULL;
    p->dii = p->dcc = NULL;
    
    i = 0;
    p->hpp = &p->hbags[i];
    p->dpp = &p->dbags[i++];
    UC(comm_bags_ini(PINNED, NONE, sizeof(Particle), capacity, /**/ p->hpp, p->dpp));
    
    if (ids) {
        p->hii = &p->hbags[i];
        p->dii = &p->dbags[i++];
        UC(comm_bags_ini(PINNED, NONE, sizeof(int), capacity, /**/ p->hii, p->dii));
    }

    if (colors) {
        p->hcc = &p->hbags[i];
        p->dcc = &p->dbags[i++];
        UC(comm_bags_ini(PINNED, NONE, sizeof(int), capacity, /**/ p->hcc, p->dcc));
    }    

    p->nbags = i;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));
}

void dflu_comm_ini(MPI_Comm cart, /**/ DFluComm **com) {
    DFluComm *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

static int nhalocells(int3 L) {
    return 8 +                  /* corners */
        4 * (L.x + L.y + L.z) + /* edges   */
        2 * L.x * L.y +         /* faces   */
        2 * L.x * L.z +
        2 * L.y * L.z;
}

void dflu_unpack_ini(bool colors, bool ids, int3 L, int maxdensity, DFluUnpack **unpack) {
    int maxparts, i, capacity[NBAGS];
    DFluUnpack *u;

    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    
    get_capacity(L, maxdensity, /**/ capacity);

    i = 0;
    u->hpp = &u->hbags[i++];
    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(Particle), capacity, /**/ u->hpp, NULL));

    if (ids) {
        u->hii = &u->hbags[i++];
        UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ u->hii, NULL));
    }
    
    if (colors) {
        u->hcc = &u->hbags[i++];
        UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int), capacity, /**/ u->hcc, NULL));
    }
    u->nbags = i;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));

    maxparts = (int) (nhalocells(L) * maxdensity) + 1;
    Dalloc(&u->ppre, maxparts);
    if (ids)    Dalloc(&u->iire, maxparts);
    if (colors) Dalloc(&u->ccre, maxparts);

    u->opt.colors = colors;
    u->opt.ids = ids;
}
