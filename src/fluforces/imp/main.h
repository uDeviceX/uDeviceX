static void zip(const int n, const float *pp, /**/ float4 *zipped_pp, ushort4 *zipped_rr) {
    static_assert(sizeof(Particle) == 6 * sizeof(float),
                  "sizeof(Particle) != 6 * sizeof(float)");
    KL(dev::zip, (k_cnf(n)), (n, pp, zipped_pp, zipped_rr));
}


void fluforces_bulk_prepare(int n, const Cloud *c, /**/ FluForcesBulk *b) {
    if (n == 0) return;
    zip(n, c->pp, /**/ b->zipped_pp, b->zipped_rr);
    if (multi_solvent)
        b->colors = c->cc;
    else
        b->colors = NULL;
}

// tmp
static void set(PairParams *p) {
    float a[] = {adpd_b, adpd_br, adpd_r};
    float g[] = {gdpd_b, gdpd_br, gdpd_r};
    float s[3];
    for (int i = 0; i < 3; ++i)
        s[i] = sqrt(2 * kBT * g[i] / dt);
    pair_set_dpd(2, a, g, s, p);
}

void fluforces_bulk_apply(int n, const FluForcesBulk *b, const int *start, const int *count, /**/ Force *ff) {
    BCloud c;
    c.pp = b->zipped_pp;
    c.cc = b->colors;

    PairParams *par;
    UC(pair_ini(&par));

    set(par);    

    if (b->colors)
        UC(flocal_color(par, b->L, n, c, start, b->rnd, /**/ ff));
    else
        UC(flocal(par, b->L, n, c, start, b->rnd, /**/ ff));

    UC(pair_fin(par));
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
PairParams *par;
    UC(pair_ini(&par));

    set(par);    

    if (multi_solvent)
        hforces::fhalo_color(par, h->L, h->lfrags, h->rfrags, h->rndfrags, (float*)ff);
    else
        hforces::fhalo(par, h->L, h->lfrags, h->rfrags, h->rndfrags, (float*)ff);
        
    UC(pair_fin(par));
}
