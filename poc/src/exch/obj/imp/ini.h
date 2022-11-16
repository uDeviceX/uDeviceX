static void estimates(int3 L, int nfrags, int maxd, int maxpsolid, int *cap) {
    int i, e, kind;
    frag_hst::estimates(L, nfrags, maxd, /**/ cap);

    /* estimates for solid */
    const float safety = 2.;
    float factor[3] = {safety / 2, safety / 4, safety / 8}; /* face, edge, corner */

    for (i = 0; i < nfrags; ++i) {
        int d[3];
        frag_hst::i2d3(i, d);
        kind = fabs(d[0]) + fabs(d[1]) + fabs(d[2]) - 1;
        e = maxpsolid * factor[kind];
        cap[i] += e;
    }
}

static void fill(int val, int n, int *a) {
    for (int i = 0; i < n; ++i) a[i] = val;
}

static void get_cc_cap(int nw, int nfrags, int *cap) {
    fill(nw, nfrags, cap);
}

static void set_cc_counts(int nw, int nfrags, hBags *ss) {
    fill(nw, nfrags, ss->counts);
}

void eobj_pack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjPack **pack) {
    int cap[NFRAGS], cccap[NFRAGS];
    EObjPack * p;
    EMALLOC(1, pack);
    p = *pack;

    p->L = L;
    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);
    get_cc_cap(nw, NFRAGS, cccap);

    UC(emap_ini(nw, NFRAGS, cap, /**/ &p->map));

    p->hpp = &p->hbags[ID_PP];  p->hcc = &p->hbags[ID_CC];
    
    UC(comm_bags_ini(PINNED,   NONE, sizeof(Particle), cap, /**/ p->hpp, &p->dpp));
    UC(comm_bags_ini(HST_ONLY, NONE, sizeof(int),    cccap, /**/ p->hcc, NULL));

    set_cc_counts(nw, NFRAGS, p->hcc);

    p->nbags = MAX_NBAGS;
    UC(comm_buffer_ini(p->nbags, p->hbags, &p->hbuf));
}

void eobj_comm_ini(MPI_Comm cart, /**/ EObjComm **com) {
    EObjComm *c;
    EMALLOC(1, com);
    c = *com;
    
    UC(comm_ini(cart, /**/ &c->comm));
}

void eobj_unpack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjUnpack **unpack) {
    int cap[NFRAGS], cccap[NFRAGS];
    EObjUnpack *u;

    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);
    get_cc_cap(nw, NFRAGS, cccap);

    u->hpp = &u->hbags[ID_PP];  u->hcc = &u->hbags[ID_CC];

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ u->hpp, &u->dpp));
    UC(comm_bags_ini(HST_ONLY  , NONE, sizeof(int),    cccap, /**/ u->hcc, NULL));

    set_cc_counts(nw, NFRAGS, u->hcc);

    u->nbags = MAX_NBAGS;
    UC(comm_buffer_ini(u->nbags, u->hbags, &u->hbuf));
}

void eobj_packf_ini(int3 L, int maxd, int maxpsolid, EObjPackF **pack) {
    int cap[NFRAGS];
    EObjPackF *p;

    EMALLOC(1, pack);
    p = *pack;
    
    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Force), cap, /**/ &p->hff, &p->dff));
}

void eobj_commf_ini(MPI_Comm cart, /**/ EObjCommF **com) {
    EObjCommF *c;
    EMALLOC(1, com);
    c = *com;
    UC(comm_ini(cart, /**/ &c->comm));
}

void eobj_unpackf_ini(int3 L, int maxd, int maxpsolid, EObjUnpackF **unpack) {
    int cap[NFRAGS];
    EObjUnpackF *u;

    EMALLOC(1, unpack);
    u = *unpack;
    
    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Force), cap, /**/ &u->hff, &u->dff));
}
