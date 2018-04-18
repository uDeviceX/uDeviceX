void eflu_pack_fin(EFluPack *p) {
    for (int i = 0; i < NFRAGS; ++i) {
        Dfree(p->bcc.d[i]);
        Dfree(p->bss.d[i]);
        Dfree(p->fss.d[i]);
        Dfree(p->bii.d[i]);
    }
    
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &p->hpp, &p->dpp));
    if (p->opt.colors)
        UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &p->hcc, &p->dcc));

    UC(comm_bags_fin(PINNED_HST, NONE, /**/ &p->hfss, NULL));

    Dfree(p->counts_dev);
    EFREE(p);
}

void eflu_comm_fin(EFluComm *c) {
    UC(comm_fin(/**/ c->pp));
    UC(comm_fin(/**/ c->fss));
    if (c->opt.colors)
        UC(comm_fin(/**/ c->cc));
    EFREE(c);
}

void eflu_unpack_fin(EFluUnpack *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &u->hpp, &u->dpp));
    if (u->opt.colors)
        UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &u->hcc, &u->dcc));

    UC(comm_bags_fin(PINNED_HST, NONE, /**/ &u->hfss, &u->dfss));
    EFREE(u);
}

