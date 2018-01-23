void dflu_pack_fin(DFluPack *p) {
    UC(dmap_fin(NFRAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED, NONE, /**/ &p->hpp, &p->dpp));

    if (global_ids)    UC(comm_bags_fin(PINNED, NONE, /**/ &p->hii, &p->dii));
    if (multi_solvent) UC(comm_bags_fin(PINNED, NONE, /**/ &p->hcc, &p->dcc));

    UC(efree(p));
}

void dflu_comm_fin(DFluComm *c) {
    UC(comm_fin(c->pp));

    if (global_ids)    UC(comm_fin(c->ii));
    if (multi_solvent) UC(comm_fin(c->cc));

    UC(efree(c));
}

void dflu_unpack_fin(DFluUnpack *u) {
    UC(comm_bags_fin(HST_ONLY, NONE, &u->hpp, NULL));
    if (global_ids)    UC(comm_bags_fin(HST_ONLY, NONE, &u->hii, NULL));
    if (multi_solvent) UC(comm_bags_fin(HST_ONLY, NONE, &u->hcc, NULL));

    CC(d::Free(u->ppre));
    if (global_ids)    CC(d::Free(u->iire));
    if (multi_solvent) CC(d::Free(u->ccre));

    UC(efree(u));
}
