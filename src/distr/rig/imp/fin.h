void fin(Pack *p) {
    fin_map(NBAGS, /**/ &p->map);
    fin(PINNED, DEV_ONLY, /**/ &p->hipp, &p->dipp);
    fin(PINNED, DEV_ONLY, /**/ &p->hss, &p->dss);
}

void fin(Comm *c) {
    fin(&c->pp);
}

void fin(Unpack *u) {
    fin(HST_ONLY, NONE, &u->hipp, NULL);
    fin(HST_ONLY, NONE, &u->hss, NULL);
}
