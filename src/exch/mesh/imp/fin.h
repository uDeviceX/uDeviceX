void fin(Pack *p) {
    fin_map(NFRAGS, /**/ &p->map);
    fin(PINNED, NONE, /**/ &p->hpp, &p->dpp);
    CC(d::Free(p->minext));
    CC(d::Free(p->maxext));
}

void fin(Comm *c) {
    fin(&c->pp);
}

void fin(Unpack *u) {
    fin(PINNED_DEV, NONE, /**/ &u->hpp, &u->dpp);
}



void fin(PackM *p) {
    fin(PINNED, NONE, /**/ &p->hmm, &p->dmm);
}

void fin(CommM *c) {
    fin(&c->mm);
}

void fin(UnpackM *u) {
    fin(PINNED_DEV, NONE, /**/ &u->hmm, &u->dmm);
}
