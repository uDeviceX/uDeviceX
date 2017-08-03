namespace x {
void ini(/*io*/ basetags::TagGen *g) {
    cnt = -1; /* TODO: */
    ini_ticketcom(&tc);
    ini_ticketr(&tr);
    ini_tickettags(g, &tt);
    ini_ticketpack(&tp);
    ini_ticketpinned(&ti);
    rex::ini();
}

void fin() {
    rex::fin();
    fin_ticketcom(tc);
    fin_ticketpack(tp);
    fin_ticketpinned(ti);
}

static void post(TicketCom tc, TicketR tr, x::TicketTags t, std::vector<ParticlesWrap> w, int nw) {
    bool packingfailed;
    dSync();
    if (cnt == 0) rex::postrecvC(tc.cart, tc.ranks, tr.tags, t);
    else          rex::s::waitC();

    rex::post_count(tp);
    packingfailed = rex::post_check();
    if (packingfailed) {
        rex::local_resize();
        rex::post_resize();
        rex::pack_clear(nw, tp);
        rex::scanA(w, nw, tp);
        rex::scanB(w, tp);
        rex::pack_attempt(w, tp);
        dSync();
    }
    rex::local_resize();
    rex::postrecvA(tc.cart, tc.ranks, tr.tags, t);

    if (cnt == 0) rex::postrecvP(tc.cart, tc.ranks, tr.tags, t);
    else          rex::s::waitP();
    rex::post_p(tc.cart, tc.ranks, t, tp);
}

static void rex0(std::vector<ParticlesWrap> w, int nw) {
    cnt++;
    rex::pack_clear(nw, tp);
    rex::scanA(w, nw, tp);
    rex::scanB(w, tp);    
    rex::pack_attempt(w, tp);
    post(tc, tr, tt, w, nw);
    rex::r::waitC();
    rex::r::waitP();
    rex::recv_p(tc.cart, tc.ranks, tr.tags, tt);
    if (cnt) rex::s::waitA();
    rex::halo(); /* fsi::halo(); */
    rex::postrecvP(tc.cart, tc.ranks, tr.tags, tt);
    rex::post_f(tc.cart, tc.ranks, tt);
    rex::recv_copy_bags();
    rex::r::waitA();
    rex::recv_f(w, tp);
}

void rex(std::vector<ParticlesWrap> w) {
    int nw;
    nw = w.size();
    if (nw) rex0(w, nw);
}

}
