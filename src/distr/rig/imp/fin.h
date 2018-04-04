void drig_pack_fin(DRigPack *p) {
    UC(dmap_fin(NBAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED, DEV_ONLY, /**/ &p->hipp, &p->dipp));
    UC(comm_bags_fin(PINNED, DEV_ONLY, /**/ &p->hss, &p->dss));
    EFREE(p);
}

void drig_comm_fin(DRigComm *c) {
    UC(comm_fin(c->ipp));
    UC(comm_fin(c->ss));
    EFREE(c);
}

void drig_unpack_fin(DRigUnpack *u) {
    UC(comm_bags_fin(HST_ONLY, NONE, &u->hipp, NULL));
    UC(comm_bags_fin(HST_ONLY, NONE, &u->hss, NULL));
    EFREE(u);
}
