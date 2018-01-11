void fin(Pack *p) {
    UC(dmap_fin(NBAGS, /**/ &p->map));
    UC(bags_fin(PINNED, DEV_ONLY, /**/ &p->hipp, &p->dipp));
    UC(bags_fin(PINNED, DEV_ONLY, /**/ &p->hss, &p->dss));
}

void fin(Comm *c) {
    fin(&c->ipp);
    fin(&c->ss);
}

void fin(Unpack *u) {
    UC(bags_fin(HST_ONLY, NONE, &u->hipp, NULL));
    UC(bags_fin(HST_ONLY, NONE, &u->hss, NULL));
}
