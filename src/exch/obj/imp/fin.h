static void fin_map(int nfrags, Map *map) {
    CC(d::Free(map->counts));
    CC(d::Free(map->starts));
    CC(d::Free(map->offsets));

    for (int i = 0; i < nfrags; ++i)
        CC(d::Free(map->ids[i]));
}

void fin(Pack *p) {
    fin_map(NFRAGS, /**/ &p->map);
    fin(PINNED, NONE, /**/ &p->hpp, &p->dpp);
}

void fin(Comm *c) {
    fin(&c->pp);
    fin(&c->ff);
}

void fin(Unpack *u) {
    fin(PINNED_DEV, NONE, /**/ &u->hpp, &u->dpp);
}

void fin(PackF *p) {
    fin(PINNED_DEV, NONE, /**/ &p->hff, &p->dff);
}

void fin(UnpackF *u) {
    fin(PINNED_DEV, NONE, /**/ &u->hff, &u->dff);
}
