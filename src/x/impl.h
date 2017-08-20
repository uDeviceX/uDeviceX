namespace x {
static void pre(std::vector<ParticlesWrap> w, int nw) {
    using namespace rex;
    cnt++;
    clear(nw, tp);
    scanA(w, nw, tp);
    copy_offset(nw, tp, ti);
    scanB(nw, tp);
    copy_tstarts(tp, ti);
    pack(w, nw, tp, buf);
}

static void send() {
    using namespace rex;
    copy_count(ti, send_counts);
    recvF(tc.cart, tc.ranks, tr.tags, tt, send_counts);
    dSync();
}

static void rex0(std::vector<ParticlesWrap> w, int nw) {
    using namespace rex;
    pre(w, nw);
    dSync();
    send();

    /** C **/
    recvC(tc.cart, tc.ranks, tr.tags, tt);
    sendC(tc.cart, tc.ranks, tt, send_counts);
    s::waitC();
    r::waitC();

    /** P **/
    recvP(tc.cart, tc.ranks, tr.tags, tt);
    sendP(tc.cart, tc.ranks, tt, ti, buf_pi, send_counts);
    s::waitP();
    r::waitP();

    if (cnt) s::waitA();
    halo(); /* fsi::halo(); */
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
