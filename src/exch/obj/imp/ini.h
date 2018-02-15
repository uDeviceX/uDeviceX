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

void eobj_pack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjPack **pack) {
    int cap[NFRAGS];
    EObjPack * p;
    UC(emalloc(sizeof(EObjPack), (void**) pack));
    p = *pack;

    p->L = L;
    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(emap_ini(nw, NFRAGS, cap, /**/ &p->map));
    UC(comm_bags_ini(PINNED, NONE, sizeof(Particle), cap, /**/ &p->hpp, &p->dpp));
}

void eobj_comm_ini(MPI_Comm cart, /**/ EObjComm **com) {
    EObjComm *c;
    UC(emalloc(sizeof(EObjComm), (void**) com));
    c = *com;
    
    UC(comm_ini(cart, /**/ &c->pp));
    UC(comm_ini(cart, /**/ &c->ff));
}

void eobj_unpack_ini(int3 L, int maxd, int maxpsolid, EObjUnpack **unpack) {
    int cap[NFRAGS];
    EObjUnpack *u;

    UC(emalloc(sizeof(EObjUnpack), (void**) unpack));
    u = *unpack;

    u->L = L;
    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ &u->hpp, &u->dpp));
}

void eobj_packf_ini(int3 L, int maxd, int maxpsolid, EObjPackF **pack) {
    int cap[NFRAGS];
    EObjPackF *p;

    UC(emalloc(sizeof(EObjPackF), (void**) pack));
    p = *pack;
    
    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Force), cap, /**/ &p->hff, &p->dff));
}

void eobj_unpackf_ini(int3 L, int maxd, int maxpsolid, EObjUnpackF **unpack) {
    int cap[NFRAGS];
    EObjUnpackF *u;

    UC(emalloc(sizeof(EObjUnpackF), (void**) unpack));
    u = *unpack;
    
    estimates(L, NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Force), cap, /**/ &u->hff, &u->dff));
}
