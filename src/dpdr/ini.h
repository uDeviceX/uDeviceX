void ini_tcomm(MPI_Comm cart, /**/ TicketCom *t) {
    int coordsneighbor[3];

    for (int i = 0; i < 26; ++i) {
        const int d[3] = {(i     + 2) % 3 - 1,
                          (i / 3 + 2) % 3 - 1,
                          (i / 9 + 2) % 3 - 1};
        
        t->recv_tags[i] = (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

        for (int c = 0; c < 3; ++c) coordsneighbor[c] = m::coords[c] + d[c];
        MC(l::m::Cart_rank(cart, coordsneighbor, t->dstranks + i));

        MC(l::m::Comm_dup(cart, /**/ t->cart));
    }
}
