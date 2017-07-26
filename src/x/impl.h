namespace x {
void ini(/*io*/ basetags::TagGen *tg) {
    cnt = -1; /* TODO: */
    ini_ticketcom(&tc);
    ini_ticketr(&tr);
    ini_tickettags(tg, &tt);
    ini_ticketpack(&tp);
    rex::ini();
}

void fin() {
    rex::fin();
    fin_ticketcom(tc);
    fin_ticketpack(tp);
}

static void post(TicketCom tc, TicketR tr, x::TicketTags t, std::vector<ParticlesWrap> w) {
    bool packingfailed;
    dSync();
    if (cnt == 0) rex::_postrecvC(tc.cart, tc.ranks, tr.tags, t);
    else          rex::post_waitC();
    packingfailed = rex::post_pre(tp);
    if (packingfailed) {
        rex::post_resize();
        rex::adjust_packbuffers();
        rex::pack_clear(tp);
        rex::scanA(w, tp);
        rex::scanB(w, tp);
        rex::pack_attempt(w, tp);
        dSync();
    }
    rex::local_resize();
    rex::postrecvA(tc.cart, tc.ranks, tr.tags, t);

    if (cnt == 0) rex::postrecvP(tc.cart, tc.ranks, tr.tags, t);
    else          rex::post_waitP();
    rex::post_p(tc.cart, tc.ranks, t, tp);
}

static void rex0(std::vector<ParticlesWrap> w, int nw) {
    cnt++;
    rex::pack_p(nw, tp);
    rex::pack_clear(tp);
    rex::scanA(w, tp);
    rex::scanB(w, tp);    
    rex::pack_attempt(w, tp);
    post(tc, tr, tt, w);
    rex::recv_p(tc.cart, tc.ranks, tr.tags, tt);
    if (cnt) rex::halo_wait();
    rex::halo(); /* fsi::halo(); */
    rex::postrecvP(tc.cart, tc.ranks, tr.tags, tt);
    rex::post_f(tc.cart, tc.ranks, tt);
    rex::recv_f(w, tp);
}

void rex(std::vector<ParticlesWrap> w) {
    int nw;
    nw = w.size();
    if (nw) rex0(w, nw);
}

}
