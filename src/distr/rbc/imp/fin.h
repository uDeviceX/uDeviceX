void fin(Pack *p) {
    UC(dmap_fin(NBAGS, /**/ &p->map));
    UC(bags_fin(PINNED, DEV_ONLY, /**/ &p->hpp, &p->dpp));
    CC(d::Free(p->minext));
    CC(d::Free(p->maxext));

    if (rbc_ids) {
        UC(dmap_fin_host(NBAGS, /**/ &p->hmap));
        UC(bags_fin(HST_ONLY, HST_ONLY, /**/ &p->hii, NULL));
    }
}

void fin(Comm *c) {
    fin(&c->pp);
    if (rbc_ids)
        fin(&c->ii);
}

void fin(Unpack *u) {
    UC(bags_fin(HST_ONLY, NONE, /**/ &u->hpp, NULL));
    if (rbc_ids)
        UC(bags_fin(HST_ONLY, NONE, /**/ &u->hii, NULL));
}
