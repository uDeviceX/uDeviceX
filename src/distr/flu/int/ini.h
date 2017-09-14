void ini(float maxdensity, Pack *p) {
    alloc_map(maxdensity, /**/ &p->map);
    ini_pinned_no_bulk(sizeof(Particle), maxdensity, /**/ &p->hpp, &p->dpp);

    if (global_ids)
        ini_pinned_no_bulk(sizeof(int), maxdensity, /**/ &p->hii, &p->dii);

    if (multi_solvent)
        ini_pinned_no_bulk(sizeof(int), maxdensity, /**/ &p->hcc, &p->dcc);
}

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c) {
    ini(comm, /*io*/ tg, /**/ &c->pp);
    if (global_ids)    ini(comm, /*io*/ tg, /**/ &c->ii);
    if (multi_solvent) ini(comm, /*io*/ tg, /**/ &c->cc);
}

void ini(Unpack *u);
