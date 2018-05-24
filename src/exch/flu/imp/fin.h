void eflu_pack_fin(EFluPack *p) {
    for (int i = 0; i < NFRAGS; ++i) {
        Dfree(p->bcc.d[i]);
        Dfree(p->bss.d[i]);
        Dfree(p->fss.d[i]);
        Dfree(p->bii.d[i]);
    }
    
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ p->hpp, p->dpp));
    if (p->hcc)
        UC(comm_bags_fin(PINNED_DEV, NONE, /**/ p->hcc, p->dcc));

    UC(comm_bags_fin(PINNED_HST, NONE, /**/ p->hfss, NULL));
    UC(comm_buffer_fin(p->hbuf));

    Dfree(p->counts_dev);
    EFREE(p);
}

void eflu_comm_fin(EFluComm *c) {
    UC(comm_fin(/**/ c->comm));
    EFREE(c);
}

void eflu_unpack_fin(EFluUnpack *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ u->hpp, u->dpp));
    if (u->hcc)
        UC(comm_bags_fin(PINNED_DEV, NONE, /**/ u->hcc, u->dcc));

    UC(comm_bags_fin(PINNED_HST, NONE, /**/ u->hfss, u->dfss));
    UC(comm_buffer_fin(u->hbuf));
    EFREE(u);
}

