void emesh_pack_fin(EMeshPack *p) {
    UC(emap_fin(NFRAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED, NONE, /**/ &p->hpp, &p->dpp));
    CC(d::Free(p->minext));
    CC(d::Free(p->maxext));
    EFREE(p);
}

void emesh_comm_fin(EMeshComm *c) {
    UC(comm_fin(c->pp));
    EFREE(c);
}

void emesh_unpack_fin(EMeshUnpack *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &u->hpp, &u->dpp));
    EFREE(u);
}

/* Momentum struct */

static void fin_map(/**/ MMap *map) {
    CC(d::Free(map->cc));
    CC(d::Free(map->ss));
    CC(d::Free(map->subids));
}

void emesh_packm_fin(EMeshPackM *p) {
    UC(comm_bags_fin(PINNED, NONE, /**/ &p->hmm, &p->dmm));
    UC(comm_bags_fin(PINNED, NONE, /**/ &p->hii, &p->dii));

    for (int i = 0; i < NFRAGS; ++i)
        UC(fin_map(&p->maps[i]));

    CC(d::FreeHost(p->cchst));
    EFREE(p);
}

void emesh_commm_fin(EMeshCommM *c) {
    UC(comm_fin(c->mm));
    UC(comm_fin(c->ii));
    EFREE(c);
}

void emesh_unpackm_fin(EMeshUnpackM *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &u->hmm, &u->dmm));
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &u->hii, &u->dii));
    EFREE(u);
}
