namespace x {

static void ini_ticketcom0(MPI_Comm cart, /**/ int ranks[26]) {
    enum {X, Y, Z};
    int i, c;
    int ne[3], d[3];
    for (i = 0; i < 26; ++i) {
        d[X] = (i     + 2) % 3 - 1;
        d[Y] = (i / 3 + 2) % 3 - 1;
        d[Z] = (i / 9 + 2) % 3 - 1;
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

void ini(/*io*/ basetags::TagGen *tg) {
    ini_ticketcom(&tc);
    rex::ini(tg);
}

void fin() {
    rex::fin();
    fin_ticketcom(tc);
}

static void post(TicketCom t, std::vector<ParticlesWrap> w) {
    rex::post_p(tc.cart, tc.ranks, w);
}

static void rex0(std::vector<ParticlesWrap> w, int nw) {
    rex::pack_p(nw);
    rex::_pack_attempt(w);
    post(tc, w);
    rex::recv_p(tc.cart, tc.ranks);
    rex::halo(); /* fsi::halo(); */
    rex::_postrecvP(tc.cart, tc.ranks);
    rex::post_f(tc.cart, tc.ranks);
    rex::recv_f(w);
}

void rex(std::vector<ParticlesWrap> w) {
    int nw;
    nw = w.size();
    if (nw) rex0(w, nw);
}

}
