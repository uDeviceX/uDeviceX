void ini(float maxdensity, Pack *p) {
    alloc_map(maxdensity, /**/ &p->map);
    ini_pinned_no_bulk(sizeof(Particle), maxdensity, /**/ &p->hpp, &p->dpp);
    if (global_ids)    ini_pinned_no_bulk(sizeof(int), maxdensity, /**/ &p->hii, &p->dii);
    if (multi_solvent) ini_pinned_no_bulk(sizeof(int), maxdensity, /**/ &p->hcc, &p->dcc);
}

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c) {
    ini(comm, /*io*/ tg, /**/ &c->pp);
    if (global_ids)    ini(comm, /*io*/ tg, /**/ &c->ii);
    if (multi_solvent) ini(comm, /*io*/ tg, /**/ &c->cc);
}

static int nhalocells() {
    return 8 +               /* corners */
        4 * (XS + YS + ZS) + /* edges   */
        2 * XS * YS +        /* faces   */
        2 * XS * ZS +
        2 * YS * ZS;
}

void ini(float maxdensity, Unpack *u) {
    ini_no_bulk(sizeof(Particle), maxdensity, /**/ &u->hpp);
    if (global_ids)    ini_no_bulk(sizeof(int), maxdensity, /**/ &u->hii);
    if (multi_solvent) ini_no_bulk(sizeof(int), maxdensity, /**/ &u->hcc);

    int maxparts = (int) (nhalocells() * maxdensity) + 1;
    CC(d::Malloc((void**) &u->ppre, maxparts * sizeof(Particle)));
    if (global_ids)    CC(d::Malloc((void**) &u->iire, maxparts * sizeof(int)));
    if (multi_solvent) CC(d::Malloc((void**) &u->ccre, maxparts * sizeof(int)));
}
