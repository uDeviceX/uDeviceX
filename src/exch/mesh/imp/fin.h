void fin(Pack *p) {
    fin_map(NFRAGS, /**/ &p->map);
    fin(PINNED, NONE, /**/ &p->hpp, &p->dpp);
    CC(d::Free(p->minext));
    CC(d::Free(p->maxext));
}

void fin(Comm *c) {
    fin(&c->pp);
}

void fin(Unpack *u) {
    fin(PINNED_DEV, NONE, /**/ &u->hpp, &u->dpp);
}

/* Momentum struct */

static void fin_map(/**/ MMap *map) {
    CC(d::Free(map->cc));
    CC(d::Free(map->ss));
    CC(d::Free(map->subids));
}

void fin(PackM *p) {
    fin(PINNED, NONE, /**/ &p->hmm, &p->dmm);
    fin(PINNED, NONE, /**/ &p->hii, &p->dii);

    for (int i = 0; i < NFRAGS; ++i)
        fin_map(&p->maps[i]);

    CC(d::FreeHost(p->cchst));
}

void fin(CommM *c) {
    fin(&c->mm);
    fin(&c->ii);
}

void fin(UnpackM *u) {
    fin(PINNED_DEV, NONE, /**/ &u->hmm, &u->dmm);
    fin(PINNED_DEV, NONE, /**/ &u->hii, &u->dii);
}
