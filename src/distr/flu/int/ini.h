void ini(float maxdensity, Pack *p) {
    alloc_map(maxdensity, /**/ &p->map);
    ini_pinned_no_bulk(sizeof(Particle), maxdensity, /**/ &p->hpp, &p->dpp);

    if (global_ids)
        ini_pinned_no_bulk(sizeof(int), maxdensity, /**/ &p->hii, &p->dii);

    if (multi_solvent)
        ini_pinned_no_bulk(sizeof(int), maxdensity, /**/ &p->hcc, &p->dcc);
}

void ini(Comm *c);
void ini(Unpack *u);
