void dflu_pack_fin(DFluPack *p) {
    UC(dmap_fin(NFRAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED, NONE, /**/ p->hpp, p->dpp));

    if (p->hii) UC(comm_bags_fin(PINNED, NONE, /**/ p->hii, p->dii));
    if (p->hcc) UC(comm_bags_fin(PINNED, NONE, /**/ p->hcc, p->dcc));

    UC(comm_buffer_fin(p->hbuf));
    EFREE(p);
}

void dflu_comm_fin(DFluComm *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void dflu_unpack_fin(DFluUnpack *u) {
    UC(comm_bags_fin(HST_ONLY, NONE, u->hpp, NULL));

    if (u->opt.ids)    UC(comm_bags_fin(HST_ONLY, NONE, u->hii, NULL));
    if (u->opt.colors) UC(comm_bags_fin(HST_ONLY, NONE, u->hcc, NULL));

    UC(comm_buffer_fin(u->hbuf));
    
    Dfree(u->ppre);
    if (u->opt.ids)    Dfree(u->iire);
    if (u->opt.colors) Dfree(u->ccre);

    EFREE(u);
}
