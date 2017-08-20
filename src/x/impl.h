namespace x {
static void pre(std::vector<ParticlesWrap> w, int nw) {
    using namespace rex;
    cnt++;
    clear(nw, tp);
    scanA(w, nw, tp);
    scanB(nw, tp);
    copy_offset(nw, tp, ti);
    copy_starts(tp, ti);
    pack(w, nw, tp, buf);
}

static void send() {
    using namespace rex;
    recvF(tc.cart, tc.ranks, tr.tags, tt, ti.counts);
    dSync();
}

static void rex0(std::vector<ParticlesWrap> w, int nw) {
    using namespace rex;
    pre(w, nw);
    dSync();
    send();

    /** C **/
    recvC(tc.cart, tc.ranks, tr.tags, tt, recv_counts);
    sendC(tc.cart, tc.ranks, tt, ti.counts);
    s::waitC();
    r::waitC();

    /** P **/
    recvP(tc.cart, tc.ranks, tr.tags, tt, recv_counts);
    sendP(tc.cart, tc.ranks, tt, ti, buf_pi, ti.counts);
    s::waitP();
    r::waitP();

    if (cnt) s::waitA();
    halo(recv_counts); /* fsi::halo(); */
    dSync();
    sendF(tc.cart, tc.ranks, tt, recv_counts); /* (sic) */
    r::waitA();
    unpack(w, nw, tp);
}

void rex(std::vector<ParticlesWrap> w) {
    int nw;
    nw = w.size();
    if (nw) rex0(w, nw);
}

}
