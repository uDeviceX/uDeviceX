static void estimates(int nfrags, int maxd, int maxpsolid, int *cap) {
    int i, e, kind;
    frag_estimates(nfrags, maxd, /**/ cap);

    /* estimates for solid */
    const float safety = 2.;
    float factor[3] = {safety / 2, safety / 4, safety / 8}; /* face, edge, corner */

    for (i = 0; i < nfrags; ++i) {
        int d[] = frag_i2d3(i);
        kind = fabs(d[0]) + fabs(d[1]) + fabs(d[2]) - 1;
        e = maxpsolid * factor[kind];
        cap[i] += e;
    }
}

void ini(int nw, int maxd, int maxpsolid, Pack *p) {
    int cap[NFRAGS];
    estimates(NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(ini_map(nw, NFRAGS, cap, /**/ &p->map));
    UC(bags_ini(PINNED, NONE, sizeof(Particle), cap, /**/ &p->hpp, &p->dpp));
}

void ini(MPI_Comm comm, /**/ Comm *c) {
    UC(comm_ini(comm, /**/ &c->pp));
    UC(comm_ini(comm, /**/ &c->ff));
}

void ini(int maxd, int maxpsolid, Unpack *u) {
    int cap[NFRAGS];
    estimates(NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ &u->hpp, &u->dpp));
}

void ini(int maxd, int maxpsolid, PackF *p) {
    int cap[NFRAGS];
    estimates(NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(bags_ini(PINNED_DEV, NONE, sizeof(Force), cap, /**/ &p->hff, &p->dff));
}

void ini(int maxd, int maxpsolid, UnpackF *u) {
    int cap[NFRAGS];
    estimates(NFRAGS, maxd, maxpsolid, /**/ cap);

    UC(bags_ini(PINNED_DEV, NONE, sizeof(Force), cap, /**/ &u->hff, &u->dff));
}
