void emesh_pack_fin(EMeshPack *p) {
    UC(emap_fin(NFRAGS, /**/ &p->map));
    UC(bags_fin(PINNED, NONE, /**/ &p->hpp, &p->dpp));
    CC(d::Free(p->minext));
    CC(d::Free(p->maxext));
    UC(efree(p));
}

void emesh_comm_fin(EMeshComm *c) {
    UC(comm_fin(&c->pp));
    UC(efree(c));
}

void emesh_unpack_fin(EMeshUnpack *u) {
    UC(bags_fin(PINNED_DEV, NONE, /**/ &u->hpp, &u->dpp));
    UC(efree(u));
}

/* Momentum struct */

static void fin_map(/**/ MMap *map) {
    CC(d::Free(map->cc));
    CC(d::Free(map->ss));
    CC(d::Free(map->subids));
}

void emesh_packm_fin(EMeshPackM *p) {
    UC(bags_fin(PINNED, NONE, /**/ &p->hmm, &p->dmm));
    UC(bags_fin(PINNED, NONE, /**/ &p->hii, &p->dii));

    for (int i = 0; i < NFRAGS; ++i)
        fin_map(&p->maps[i]);

    CC(d::FreeHost(p->cchst));
    UC(efree(p));
}

void emesh_commm_fin(EMeshCommM *c) {
    UC(comm_fin(&c->mm));
    UC(comm_fin(&c->ii));
    UC(efree(c));
}

void emesh_unpackm_fin(EMeshUnpackM *u) {
    UC(bags_fin(PINNED_DEV, NONE, /**/ &u->hmm, &u->dmm));
    UC(bags_fin(PINNED_DEV, NONE, /**/ &u->hii, &u->dii));
    UC(efree(u));
}
