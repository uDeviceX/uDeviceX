void ini_tcomm(MPI_Comm cart, /**/ MPI_Comm *newcart, int dstranks[], int recv_tags[]) {
    int coordsneighbor[3];

    for (int i = 0; i < 26; ++i) {
        const int d[3] = {(i     + 2) % 3 - 1,
                          (i / 3 + 2) % 3 - 1,
                          (i / 9 + 2) % 3 - 1};
        
        recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

        for (int c = 0; c < 3; ++c) coordsneighbor[c] = m::coords[c] + d[c];
        MC(l::m::Cart_rank(cart, coordsneighbor, dstranks + i));
        MC(l::m::Comm_dup(cart, /**/ newcart));
    }
}

static void ini_one_trunk(const int i, const int dstrank, /**/ l::rnd::d::KISS* interrank_trunks[], bool interrank_masks[]) {
    int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};

    int coordsneighbor[3];
    for (int c = 0; c < 3; ++c)
    coordsneighbor[c] = (m::coords[c] + d[c] + m::dims[c]) % m::dims[c];

    int indx[3];
    for (int c = 0; c < 3; ++c)
    indx[c] = min(m::coords[c], coordsneighbor[c]) * m::dims[c] +
        max(m::coords[c], coordsneighbor[c]);

    int interrank_seed_base =
        indx[0] + m::dims[0] * m::dims[0] * (indx[1] + m::dims[1] * m::dims[1] * indx[2]);

    int interrank_seed_offset;

    {
        bool isplus =
            d[0] + d[1] + d[2] > 0 ||
            d[0] + d[1] + d[2] == 0 &&
            (d[0] > 0 || d[0] == 0 && (d[1] > 0 || d[1] == 0 && d[2] > 0));

        int mysign = 2 * isplus - 1;

        int v[3] = {1 + mysign * d[0], 1 + mysign * d[1], 1 + mysign * d[2]};

        interrank_seed_offset = v[0] + 3 * (v[1] + 3 * v[2]);
    }

    int interrank_seed = interrank_seed_base + interrank_seed_offset;

    interrank_trunks[i] = new l::rnd::d::KISS(390 + interrank_seed, interrank_seed + 615, 12309, 23094);
    
    if (dstrank != m::rank)
    interrank_masks[i] = min(dstrank, m::rank) == m::rank;
    else {
        int alter_ego =
            (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
        interrank_masks[i] = min(i, alter_ego) == i;
    }
}

void ini_trnd(const int dstranks[], /**/ l::rnd::d::KISS* interrank_trunks[], bool interrank_masks[]) {
    for (int i = 0; i < 26; ++i)
    ini_one_trunk(i, dstranks[i], /**/ interrank_trunks, interrank_masks);
}
