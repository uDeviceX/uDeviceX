namespace x {

static void ini_ticketcom(MPI_Comm cart, /**/ int dstranks[26]) {

}

void ini(/*io*/ basetags::TagGen *tg) {
    MC(l::m::Comm_dup(m::cart, &cart));
    rex::ini(cart, tg);
}

void fin() {
    rex::fin();
    MC(l::m::Comm_free(&cart));
}

void rex0(std::vector<ParticlesWrap> w, int nw) {
    rex::pack_p(nw);
    rex::_pack_attempt(w);
    rex::post_p(cart, w);
    rex::recv_p(cart, w);
    rex::halo(); /* fsi::halo(); */
    rex::_postrecvP(cart);
    rex::post_f(cart, w);
    rex::recv_f(w);
}

void rex(std::vector<ParticlesWrap> w) {
    int nw;
    nw = w.size();
    if (nw) rex0(w, nw);
}

}
