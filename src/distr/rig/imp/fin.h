void drig_pack_fin(DRigPack *p) {
    UC(dmap_fin(NBAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED, DEV_ONLY, /**/ p->hipp, p->dipp));
    UC(comm_bags_fin(PINNED, DEV_ONLY, /**/ p->hss, p->dss));
    UC(comm_buffer_fin(p->hbuf));
    EFREE(p);
}

void drig_comm_fin(DRigComm *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void drig_unpack_fin(DRigUnpack *u) {
    UC(comm_bags_fin(HST_ONLY, NONE, u->hipp, NULL));
    UC(comm_bags_fin(HST_ONLY, NONE, u->hss, NULL));
    UC(comm_buffer_fin(u->hbuf));
    EFREE(u);
}
