static void zip(const int n, const float *pp, /**/ float4 *zipped_pp, ushort4 *zipped_rr) {
    assert(sizeof(Particle) == 6 * sizeof(float)); /* :TODO: implicit dependency */
    KL(dev::zip, (k_cnf(n)), (n, pp, zipped_pp, zipped_rr));
}


void fluforces_bulk_prepare(int n, const Cloud *c, /**/ FluForcesBulk *b) {
    if (n == 0) return;
    zip(n, c->pp, /**/ b->zipped_pp, b->zipped_rr);
    if (multi_solvent)
        b->colors = c->cc;
}

void fluforces_bulk_apply(int n, const FluForcesBulk *b, const int *start, const int *count, /**/ Force *ff) {
    if (multi_solvent)
        flocal_color(b->zipped_pp, b->zipped_rr, b->colors, n, start, count, b->rnd, /**/ ff);
    else
        flocal(b->zipped_pp, b->zipped_rr, n, start, count, b->rnd, /**/ ff);
}


void fluforces_halo_prepare(flu::LFrag26 lfrags, flu::RFrag26 rfrags, /**/ FluForcesHalo *h) {
    h->lfrags = lfrags;
    h->rfrags = rfrags;

    for (int i = 0; i < 26; ++i) {
        h->rndfrags.d[i] = {
            .seed = rnd_get(h->trunks[i]),
            .mask = h->masks[i]
        };
    }
}

void fluforces_halo_apply(const FluForcesHalo *h, /**/ Force *ff) {
    hforces::interactions(h->lfrags, h->rfrags, h->rndfrags, (float*)ff);
}
