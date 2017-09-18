void fin(Pack *p) {
    fin_map(/**/ &p->map);
    fin(PINNED, DEV_ONLY, /**/ &p->hpp, &p->dpp);
}

void fin(Comm *c) {
    fin(&c->pp);
}

void fin(Unpack *u) {
    fin(HST_ONLY, NONE, &u->hpp, NULL);
}
