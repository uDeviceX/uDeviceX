namespace x {
void ini(/*io*/ basetags::TagGen *g) {
    cnt = -1; /* TODO: */
    ini_ticketcom(&tc);
    ini_ticketr(&tr);
    ini_tickettags(g, &tt);
    ini_ticketpack(&tp);
    ini_ticketpinned(&ti);
    mpDeviceMalloc(&packbuf);
    Palloc(&host_packbuf, MAX_PART_NUM);
    rex::ini();
}

void fin() {
    rex::fin();
    cudaFree(packbuf);
    Pfree(host_packbuf);
    fin_ticketcom(tc);
    fin_ticketpack(tp);
    fin_ticketpinned(ti);
}

static void post(std::vector<ParticlesWrap> w, int nw) {
    bool packingfailed;
    dSync();
    if (cnt == 0) rex::recvC(tc.cart, tc.ranks, tr.tags, tt);
    else          rex::s::waitC();

    rex::copy_count(ti);
    packingfailed = rex::post_check();
    if (packingfailed) {
        rex::local_resize();
        rex::post_resize();
        rex::clear(nw, tp);
        rex::scanA( w, nw, tp);
        rex::copy_offset(nw, tp, ti);
        rex::scanB(nw, tp);
        rex::copy_tstarts(tp, ti);
        rex::pack(w, nw, tp, packbuf);
        dSync();
    }
    rex::local_resize();
    rex::recvF(tc.cart, tc.ranks, tr.tags, tt);

    if (cnt == 0) rex::recvP(tc.cart, tc.ranks, tr.tags, tt);
    else          rex::s::waitP();
    rex::copy_pack(ti, packbuf);
    dSync();
    rex::sendC(tc.cart, tc.ranks, tt);
    rex::sendP(tc.cart, tc.ranks, tt, ti);
}

static void rex0(std::vector<ParticlesWrap> w, int nw) {
    cnt++;
    rex::clear(nw, tp);
    rex::scanA(w, nw, tp);
    rex::copy_offset(nw, tp, ti);
    rex::scanB(nw, tp);
    rex::copy_tstarts(tp, ti);
    rex::pack(w, nw, tp, packbuf);
    post(w, nw);
    rex::r::waitC();
    rex::r::waitP();
    rex::recvM(tc.cart, tc.ranks, tr.tags, tt);
    rex::recvC(tc.cart, tc.ranks, tr.tags, tt);
    rex::copy_state();
    if (cnt) rex::s::waitA();
    rex::halo(); /* fsi::halo(); */
    rex::recvP(tc.cart, tc.ranks, tr.tags, tt);
    dSync();
    rex::sendF(tc.cart, tc.ranks, tt);
    rex::copy_ff();
    rex::r::waitA();
    rex::unpack(w, nw, tp);
}

void rex(std::vector<ParticlesWrap> w) {
    int nw;
    nw = w.size();
    if (nw) rex0(w, nw);
}

}
