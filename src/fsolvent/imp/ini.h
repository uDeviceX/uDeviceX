void ini(int maxp, /**/ BulkData **bd) {
    BulkData *b = new BulkData;
    Dalloc(&b->zipped_pp, 2 * maxp);
    Dalloc(&b->zipped_rr,     maxp);
    b->rnd = new rnd::KISS(0, 0, 0, 0);
    b->colors = NULL;
    *bd = b;
}

static int is_plus(const int d[3]) {
    enum {X, Y, Z};
    return d[X] + d[Y] + d[Z] > 0 ||
        d[X] + d[Y] + d[Z] == 0 && (d[X] > 0 || d[X] == 0 && (d[Y] > 0 || d[Y] == 0 && d[Z] > 0));
}

static void get_interrank_infos(int fid, /**/ rnd::KISS* trunks[], bool masks[]) {
    int coordsneighbor[3], c, indx[3], dstrank;
    int seed, seed_base, seed_offset; 
    int d[3] = frag_i2d3(fid);
    
    for (c = 0; c < 3; ++c)
        coordsneighbor[c] = (m::coords[c] + d[c] + m::dims[c]) % m::dims[c];

    MC(m::Cart_rank(m::cart, m::coords, /**/ &dstrank));

    for ( c = 0; c < 3; ++c)
        indx[c] = min(m::coords[c], coordsneighbor[c]) * m::dims[c] +
            max(m::coords[c], coordsneighbor[c]);

    seed_base = indx[0] + m::dims[0] * m::dims[0] * (indx[1] + m::dims[1] * m::dims[1] * indx[2]);

    {
        int mysign = 2 * is_plus(d) - 1;

        int v[3] = {1 + mysign * d[0], 1 + mysign * d[1], 1 + mysign * d[2]};

        seed_offset = v[0] + 3 * (v[1] + 3 * v[2]);
    }

    seed = seed_base + seed_offset;

    trunks[fid] = new rnd::KISS(390 + seed, seed + 615, 12309, 23094);
    
    if (dstrank != m::rank)
        masks[fid] = min(dstrank, m::rank) == m::rank;
    else {
        int alter_ego = frag_anti(fid);
        masks[fid] = min(fid, alter_ego) == fid;
    }
}

void ini(/**/ HaloData **hd) {
    HaloData *h = new HaloData;
    for (int i = 0; i < 26; ++i)
        get_interrank_infos(i, /**/ h->trunks, h->masks);
    *hd = h;
}
