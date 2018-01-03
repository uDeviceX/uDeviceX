void ini(int maxp, /**/ BulkData **bd) {
    BulkData *b = new BulkData;
    Dalloc(&b->zipped_pp, 2 * maxp);
    Dalloc(&b->zipped_rr,     maxp);
    UC(rnd_ini(0, 0, 0, 0, /**/ &b->rnd));
    b->colors = NULL;
    *bd = b;
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
    int d[3] = frag_i2d3(fid);

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
        int alter_ego = frag_anti(fid);
        masks[fid] = min(fid, alter_ego) == fid;
    }

    // msg_print("%d %d %d [%d %d %d]", fid, seed, masks[fid],
    //     frag_i2dx(fid), frag_i2dy(fid), frag_i2dz(fid));
}

void ini(MPI_Comm cart, /**/ HaloData **hd) {
    HaloData *h = new HaloData;
    for (int i = 0; i < 26; ++i)
        get_interrank_infos(cart, i, /**/ h->trunks, h->masks);
    *hd = h;
}
