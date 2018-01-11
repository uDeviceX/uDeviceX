void fin(Pack *p) {
    fin_map(NFRAGS, /**/ &p->map);
    UC(bags_fin(PINNED, NONE, /**/ &p->hpp, &p->dpp));
}

void fin(Comm *c) {
    fin(&c->pp);
    fin(&c->ff);
}

void fin(Unpack *u) {
    UC(bags_fin(PINNED_DEV, NONE, /**/ &u->hpp, &u->dpp));
}

void fin(PackF *p) {
    UC(bags_fin(PINNED_DEV, NONE, /**/ &p->hff, &p->dff));
}

void fin(UnpackF *u) {
    UC(bags_fin(PINNED_DEV, NONE, /**/ &u->hff, &u->dff));
}
