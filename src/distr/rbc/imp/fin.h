void drbc_pack_fin(DRbcPack *p) {
    UC(dmap_fin(NBAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED, DEV_ONLY, /**/ p->hpp, p->dpp));
    Dfree(p->minext);
    Dfree(p->maxext);

    if (p->ids) {
        UC(dmap_fin_host(NBAGS, /**/ &p->hmap));
        UC(comm_bags_fin(HST_ONLY, HST_ONLY, /**/ p->hii, NULL));
    }
    UC(comm_buffer_fin(p->hbuf));
    EFREE(p);
}

void drbc_comm_fin(DRbcComm *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void drbc_unpack_fin(DRbcUnpack *u) {
    UC(comm_bags_fin(HST_ONLY, NONE, /**/ u->hpp, NULL));
    if (u->ids)
        UC(comm_bags_fin(HST_ONLY, NONE, /**/ u->hii, NULL));
    UC(comm_buffer_fin(u->hbuf));
    EFREE(u);
}
