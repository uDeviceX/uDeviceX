namespace x {
void ini(/*io*/ basetags::TagGen *tg) {
    ini_ticketcom(&tc);
    rex::ini(tg);
}

void fin() {
    rex::fin();
    fin_ticketcom(tc);
}

static void post(TicketCom t, std::vector<ParticlesWrap> w) {
    bool packingfailed;
    packingfailed = rex::post_pre(t.cart, t.ranks);
    if (packingfailed) {
        rex::post_resize();
        rex::_adjust_packbuffers();
        rex::_pack_attempt(w);
        dSync();
    }
    rex::post_p(t.cart, t.ranks);
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
