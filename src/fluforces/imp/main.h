static void zip(const int n, const float *pp, /**/ float4 *zpp) {
    static_assert(sizeof(Particle) == 6 * sizeof(float),
                  "sizeof(Particle) != 6 * sizeof(float)");
    KL(fluforces_dev::zip, (k_cnf(n)), (n, pp, zpp));
}


void fluforces_bulk_prepare(int n, const Cloud *c, /**/ FluForcesBulk *b) {
    if (n == 0) return;
    zip(n, c->pp, /**/ b->zipped_pp);
    if (multi_solvent)
        b->colors = c->cc;
    else
        b->colors = NULL;
}

void fluforces_bulk_apply(const PairParams *par, int n, const FluForcesBulk *b, const int *start, const int *count, /**/ Force *ff) {
    BCloud c;
    if (n == 0) return;
    c.pp = b->zipped_pp;
    c.cc = b->colors;

    if (b->colors)
        UC(flocal_color(par, b->L, n, c, start, b->rnd, /**/ ff));
    else
        UC(flocal(par, b->L, n, c, start, b->rnd, /**/ ff));
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

void fluforces_halo_apply(const PairParams *par, const FluForcesHalo *h, /**/ Force *ff) {
    if (multi_solvent)
        fhalo_apply_color(par, h->L, h->lfrags, h->rfrags, h->rndfrags, (float*)ff);
    else
        fhalo_apply(par, h->L, h->lfrags, h->rfrags, h->rndfrags, (float*)ff);
}
