void eflu_pack_fin(EFluPack *p) {
    for (int i = 0; i < NFRAGS; ++i) {
        CC(d::Free(p->bcc.d[i]));
        CC(d::Free(p->bss.d[i]));
        CC(d::Free(p->fss.d[i]));
        CC(d::Free(p->bii.d[i]));
    }
    
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &p->hpp, &p->dpp));
    if (multi_solvent)
        UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &p->hcc, &p->dcc));

    UC(comm_bags_fin(PINNED_HST, NONE, /**/ &p->hfss, NULL));

    CC(d::Free(p->counts_dev));
    UC(efree(p));
}

void eflu_comm_fin(EFluComm *c) {
    UC(comm_fin(/**/ c->pp));
    UC(comm_fin(/**/ c->fss));
    if (multi_solvent)
        UC(comm_fin(/**/ c->cc));
    UC(efree(c));
}

void eflu_unpack_fin(EFluUnpack *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &u->hpp, &u->dpp));
    if (multi_solvent)
        UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &u->hcc, &u->dcc));

    UC(comm_bags_fin(PINNED_HST, NONE, /**/ &u->hfss, &u->dfss));
    UC(efree(u));
}

