void drig_pack_fin(DRigPack *p) {
    UC(dmap_fin(NBAGS, /**/ &p->map));
    UC(bags_fin(PINNED, DEV_ONLY, /**/ &p->hipp, &p->dipp));
    UC(bags_fin(PINNED, DEV_ONLY, /**/ &p->hss, &p->dss));
    UC(efree(p));
}

void drig_comm_fin(DRigComm *c) {
    UC(comm_fin(c->ipp));
    UC(comm_fin(c->ss));
    UC(efree(c));
}

void drig_unpack_fin(DRigUnpack *u) {
    UC(bags_fin(HST_ONLY, NONE, &u->hipp, NULL));
    UC(bags_fin(HST_ONLY, NONE, &u->hss, NULL));
    UC(efree(u));
}
