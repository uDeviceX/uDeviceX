namespace x {
static void send(std::vector<ParticlesWrap> w, int nw) {
    using namespace rex;

    bool packingfailed;
    dSync();
    if (cnt == 0) recvC(tc.cart, tc.ranks, tr.tags, tt);
    else          s::waitC();

    copy_count(ti);
    packingfailed = post_check();
    if (packingfailed) {
        local_resize();
        post_resize();
        clear(nw, tp);
        scanA( w, nw, tp);
        copy_offset(nw, tp, ti);
        scanB(nw, tp);
        copy_tstarts(tp, ti);
        pack(w, nw, tp, buf);
        dSync();
    }
    local_resize();
    recvF(tc.cart, tc.ranks, tr.tags, tt);

    if (cnt == 0) recvP(tc.cart, tc.ranks, tr.tags, tt);
    else          s::waitP();
    copy_pack(ti, buf, buf_pinned);
    dSync();
    sendC(tc.cart, tc.ranks, tt);
    sendP12(tc.cart, tc.ranks, tt, ti, buf_pinned);
}

static void rex0(std::vector<ParticlesWrap> w, int nw) {
    using namespace rex;
    cnt++;
    clear(nw, tp);
    scanA(w, nw, tp);
    copy_offset(nw, tp, ti);
    scanB(nw, tp);
    copy_tstarts(tp, ti);
    pack(w, nw, tp, buf);
    send(w, nw);
    r::waitC();
    r::waitP();
    recvM(tc.cart, tc.ranks, tr.tags, tt);
    copy_hstate();
    recvC(tc.cart, tc.ranks, tr.tags, tt);
    copy_state();
    if (cnt) s::waitA();
    halo(); /* fsi::halo(); */
    recvP(tc.cart, tc.ranks, tr.tags, tt);
    dSync();
    sendF(tc.cart, tc.ranks, tt);
    copy_ff();
    r::waitA();
    unpack(w, nw, tp);
}

void rex(std::vector<ParticlesWrap> w) {
    int nw;
    nw = w.size();
    if (nw) rex0(w, nw);
}

}
