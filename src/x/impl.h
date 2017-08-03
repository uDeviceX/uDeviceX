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

static void post(std::vector<ParticlesWrap> w, int nw) {
    bool packingfailed;
    dSync();
    if (cnt == 0) rex::recvC(tc.cart, tc.ranks, tr.tags, tt);
    else          rex::s::waitC();

    rex::post_count(ti);
    packingfailed = rex::post_check();
    if (packingfailed) {
        rex::local_resize();
        rex::post_resize();
        rex::pack_clear(nw, tp);
        rex::scanA( w, nw, tp);
        rex::scanB(nw, tp, ti);
        rex::pack_attempt(w, tp, ti);
        dSync();
    }
    rex::local_resize();
    rex::recvF(tc.cart, tc.ranks, tr.tags, tt);

    if (cnt == 0) rex::recvP(tc.cart, tc.ranks, tr.tags, tt);
    else          rex::s::waitP();
    rex::sendP(tc.cart, tc.ranks, tt, ti);
}

static void rex0(std::vector<ParticlesWrap> w, int nw) {
    cnt++;
    rex::pack_clear(nw, tp);
    rex::scanA(w, nw, tp);
    rex::scanB(nw, tp, ti);
    rex::pack_attempt(w, tp, ti);
    post(w, nw);
    rex::r::waitC();
    rex::r::waitP();
    rex::recvM(tc.cart, tc.ranks, tr.tags, tt);
    if (cnt) rex::s::waitA();
    rex::halo(); /* fsi::halo(); */
    rex::recvP(tc.cart, tc.ranks, tr.tags, tt);
    rex::sendF(tc.cart, tc.ranks, tt);
    rex::recv_copy_bags();
    rex::r::waitA();
    rex::unpack(w, tp);
}

void rex(std::vector<ParticlesWrap> w) {
    int nw;
    nw = w.size();
    if (nw) rex0(w, nw);
}

}
