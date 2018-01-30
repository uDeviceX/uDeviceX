void fluforces_bulk_ini(int3 L, int maxp, /**/ FluForcesBulk **bd) {
    FluForcesBulk *b;
    UC(emalloc(sizeof(FluForcesBulk), (void**) bd));
    b = *bd;
    Dalloc(&b->zipped_pp, 2 * maxp);
    Dalloc(&b->zipped_rr,     maxp);
    UC(rnd_ini(0, 0, 0, 0, /**/ &b->rnd));
    b->colors = NULL;
    b->L = L;
}

static int is_plus(const int d[3]) {
    enum {X, Y, Z};
    return d[X] + d[Y] + d[Z] > 0 ||
        d[X] + d[Y] + d[Z] == 0 && (d[X] > 0 || d[X] == 0 && (d[Y] > 0 || d[Y] == 0 && d[Z] > 0));
}

static void get_interrank_infos(MPI_Comm cart, int fid, /**/ RNDunif* trunks[], bool masks[]) {
    int coordsneighbor[3], c, indx[3], rank, dstrank;
    int seed, seed_base, seed_offset;
    int dims[3], periods[3], coords[3];
    int d[3];
    fraghst::i2d3(fid, d);

    MC(m::Cart_get(cart, 3, dims, periods, coords));    
    MC(m::Cart_rank(cart, coords, /**/ &rank));
    
    for (c = 0; c < 3; ++c)
        coordsneighbor[c] = (coords[c] + d[c] + dims[c]) % dims[c];

    MC(m::Cart_rank(cart, coordsneighbor, /**/ &dstrank));

    for (c = 0; c < 3; ++c)
        indx[c] = min(coords[c], coordsneighbor[c]) * dims[c] +
            max(coords[c], coordsneighbor[c]);

    seed_base = indx[0] + dims[0] * dims[0] * (indx[1] + dims[1] * dims[1] * indx[2]);

    {
        int mysign = 2 * is_plus(d) - 1;
        int v[3] = {1 + mysign * d[0], 1 + mysign * d[1], 1 + mysign * d[2]};
        seed_offset = v[0] + 3 * (v[1] + 3 * v[2]);
    }

    seed = seed_base + seed_offset;

    UC(rnd_ini(390 + seed, seed + 615, 12309, 23094, /**/ &trunks[fid]));
    
    if (dstrank != rank)
        masks[fid] = min(dstrank, rank) == rank;
    else {
        int alter_ego = fraghst::frag_anti(fid);
        masks[fid] = min(fid, alter_ego) == fid;
    }
}

void fluforces_halo_ini(MPI_Comm cart, int3 L, /**/ FluForcesHalo **hd) {
    FluForcesHalo *h;
    UC(emalloc(sizeof(FluForcesHalo), (void**) hd));
    h = *hd;

    h->L = L;
    
    for (int i = 0; i < 26; ++i)
        get_interrank_infos(cart, i, /**/ h->trunks, h->masks);
}
