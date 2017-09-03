namespace x {
static void ini_ticketcom(/**/ TicketCom *t) {
    int i, c;
    int ne[3], *rank;
    for (i = 0; i < 26; ++i) {
        const int *d = frag_to_dir[i];
        for (c = 0; c < 3; ++c) ne[c] = m::coords[c] + d[c];
        rank = &(t->ranks[i]);
        MC(m::Cart_rank(m::cart, ne, /**/ rank));
    }
}

}
