namespace x {
void ini(/*io*/ basetags::TagGen *tg) {
    ini_ticketcom(&tc);
    ini_ticketr(&tr);
    rex::ini(tg);
}

void fin() {
    rex::fin();
    fin_ticketcom(tc);
}

static void post(TicketCom tc, TicketR tr, std::vector<ParticlesWrap> w) {
    bool packingfailed;
    packingfailed = rex::post_pre(tc.cart, tc.ranks, tr.tags);
    if (packingfailed) {
        rex::post_resize();
        rex::_adjust_packbuffers();
        rex::_pack_attempt(w);
        dSync();
    }
    rex::post_p(tc.cart, tc.ranks, tr.tags);
}

static void rex0(std::vector<ParticlesWrap> w, int nw) {
    rex::pack_p(nw);
    rex::_pack_attempt(w);
    post(tc, tr, w);
    rex::recv_p(tc.cart, tc.ranks, tr.tags);
    rex::halo(); /* fsi::halo(); */
    rex::_postrecvP(tc.cart, tc.ranks, tr.tags);
    rex::post_f(tc.cart, tc.ranks);
    rex::recv_f(w);
}

void rex(std::vector<ParticlesWrap> w) {
    int nw;
    nw = w.size();
    if (nw) rex0(w, nw);
}

}
