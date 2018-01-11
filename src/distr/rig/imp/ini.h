static void get_num_capacity(int maxns, /**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = maxns;
}

void drig_pack_ini(int maxns, int nv, Pack *p) {
    int numc[NBAGS];
    get_num_capacity(maxns, /**/ numc);

    UC(dmap_ini(NBAGS, numc, /**/ &p->map));

    /* one datum is here a full mesh, so bsize is nv * sizeof(Particle) */
    UC(bags_ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ &p->hipp, &p->dipp));
    
    UC(bags_ini(PINNED, DEV_ONLY, sizeof(Solid), numc, /**/ &p->hss, &p->dss));
}

void drig_comm_ini(MPI_Comm comm, /**/ Comm *c) {
    UC(comm_ini(comm, /**/ &c->ipp));
    UC(comm_ini(comm, /**/ &c->ss));
}

void drig_unpack_ini(int maxns, int nv, Unpack *u) {
    int numc[NBAGS];
    get_num_capacity(maxns, /**/ numc);

    /* one datum is here a full mesh, so bsize is nv * sizeof(Particle) */
    UC(bags_ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ &u->hipp, NULL));

    UC(bags_ini(HST_ONLY, NONE, sizeof(Solid), numc, /**/ &u->hss, NULL));
}
