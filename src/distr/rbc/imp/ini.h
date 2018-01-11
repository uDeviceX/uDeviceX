static void get_num_capacity(int maxnc, /**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = maxnc;
}

void ini(int maxnc, int nv, Pack *p) {
    int numc[NBAGS];
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

void ini(MPI_Comm comm, /**/ Comm *c) {
    UC(comm_ini(comm, /**/ &c->pp));
    if (rbc_ids)
        UC(comm_ini(comm, /**/ &c->ii));
}

void ini(int maxnc, int nv, Unpack *u) {
    int numc[NBAGS];
    get_num_capacity(maxnc, /**/ numc);

    /* one datum is here a full RBC, so bsize is nv * sizeof(Particle) */
    UC(bags_ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ &u->hpp, NULL));

    if (rbc_ids)
        UC(bags_ini(HST_ONLY, NONE, sizeof(int), numc, /**/ &u->hii, NULL));
}
