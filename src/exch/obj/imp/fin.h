void fin(Pack *p) {
    fin_map(NFRAGS, /**/ &p->map);
    fin(PINNED, NONE, /**/ &p->hpp, &p->dpp);
}

void fin(Comm *c) {
    fin(&c->pp);
    fin(&c->ff);
}

void fin(Unpack *u) {
    fin(PINNED_DEV, NONE, /**/ &u->hpp, &u->dpp);
}

void fin(PackF *p) {
    fin(PINNED_DEV, NONE, /**/ &p->hff, &p->dff);
}

void fin(UnpackF *u) {
    fin(PINNED_DEV, NONE, /**/ &u->hff, &u->dff);
}
