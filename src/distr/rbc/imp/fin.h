void drbc_pack_fin(DRbcPack *p) {
    UC(dmap_fin(NBAGS, /**/ &p->map));
    UC(bags_fin(PINNED, DEV_ONLY, /**/ &p->hpp, &p->dpp));
    CC(d::Free(p->minext));
    CC(d::Free(p->maxext));

    if (rbc_ids) {
        UC(dmap_fin_host(NBAGS, /**/ &p->hmap));
        UC(bags_fin(HST_ONLY, HST_ONLY, /**/ &p->hii, NULL));
    }
    UC(efree(p));
}

void drbc_comm_fin(DRbcComm *c) {
    UC(comm_fin(c->pp));
    if (rbc_ids)
        UC(comm_fin(c->ii));
    UC(efree(c));
}

void drbc_unpack_fin(DRbcUnpack *u) {
    UC(bags_fin(HST_ONLY, NONE, /**/ &u->hpp, NULL));
    if (rbc_ids)
        UC(bags_fin(HST_ONLY, NONE, /**/ &u->hii, NULL));
    UC(efree(u));
}
