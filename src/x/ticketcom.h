namespace x {
static void ini_ticketcom0(MPI_Comm cart, /**/ int ranks[26]) {
    int i, c;
    int ne[3], d[3];
    for (i = 0; i < 26; ++i) {
        i2d(i, /**/ d);
        for (c = 0; c < 3; ++c) ne[c] = m::coords[c] + d[c];
        MC(l::m::Cart_rank(cart, ne, ranks + i));
    }
}

static void ini_ticketcom(TicketCom *t) {
    MC(l::m::Comm_dup(m::cart, &t->cart));
    ini_ticketcom0(t->cart, t->ranks);
}

static void fin_ticketcom(TicketCom t) {
    MC(l::m::Comm_free(&t.cart));
}
}
