void fin(/**/ FluForcesBulk *b) {
    CC(d::Free(b->zipped_pp));
    CC(d::Free(b->zipped_rr));
    UC(rnd_fin(b->rnd));
    UC(efree(b));;
}

void fin(/**/ HaloData *h) {
    for (int i = 0; i < 26; ++i)
        UC(rnd_fin(h->trunks[i]));
    delete h;
}
