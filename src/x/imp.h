namespace x {
static void pre(ParticlesWrap *w, int nw) {
    using namespace rex;
    clear(nw, tp);
    scanA(w, nw, tp);
    scanB(nw, tp);
    copy_offset(nw, tp, ti);
    copy_starts(tp, ti);
    pack(w, nw, tp, buf);
}

static void rex0(ParticlesWrap *w, int nw) {
    using namespace rex;
    dSync();
    recvF(tc.cart, tc.ranks, tr.tags, tt, ti.counts);
    dSync();

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

    if (!first) s::waitA(); else first = 0;
    halo(recv_counts); /* fsi::halo(); */
    dSync();
    sendF(tc.cart, tc.ranks, tt, recv_counts); /* (sic) */
    r::waitA();
    unpack(w, nw, tp);
}

void rex(std::vector<ParticlesWrap> w0) {
    int nw;
    ParticlesWrap *w;
    
    nw = w0.size();
    w  = w0.data();
    
    if (nw) {
        pre(w, nw);
        rex0(w, nw);
    }
}

}
