void eobj_pack_fin(EObjPack *p) {
    UC(emap_fin(NFRAGS, /**/ &p->map));
    UC(comm_bags_fin(PINNED,   NONE, /**/ p->hpp, &p->dpp));
    UC(comm_bags_fin(HST_ONLY, NONE, /**/ p->hcc, NULL));
    EFREE(p);
}

void eobj_comm_fin(EObjComm *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void eobj_unpack_fin(EObjUnpack *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ u->hpp, &u->dpp));
    UC(comm_bags_fin(HST_ONLY  , NONE, /**/ u->hcc, NULL));
    EFREE(u);
}

void eobj_packf_fin(EObjPackF *p) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &p->hff, &p->dff));
    EFREE(p);
}

void eobj_commf_fin(EObjCommF *c) {
    UC(comm_fin(c->comm));
    EFREE(c);
}

void eobj_unpackf_fin(EObjUnpackF *u) {
    UC(comm_bags_fin(PINNED_DEV, NONE, /**/ &u->hff, &u->dff));
    EFREE(u);
}
