namespace rex {
static void pre(ParticlesWrap *w, int nw) {
    using namespace sub;
    clear(nw, tp);
    scanA(w, nw, tp);
    scanB(nw, tp);
    copy_offset(nw, tp, /**/ ti);
    copy_starts(tp, /**/ ti);
    pack(w, nw, tp, /**/ buf);
}

static void rex0(ParticlesWrap *w, int nw) {
    using namespace sub;
    dSync();
    recvF(tc.ranks, tr.tags, tt, ti.counts, local);
    dSync();

    /** C **/
    recvC(tc.ranks, tr.tags, tt, recv_counts);
    sendC(tc.ranks, tt, ti.counts);
    s::waitC();
    r::waitC();

    /** P **/
    recvP(tc.ranks, tr.tags, tt, recv_counts, PP_pi);
    sendP(tc.ranks, tt, ti, buf_pi, ti.counts);
    s::waitP();
    r::waitP();

    if (!first) s::waitA(); else first = 0;

    dSync();
    clear_forces(recv_counts, FF);
    if (fsiforces)     fsi::halo(PP, FF, recv_counts);
    if (contactforces) cnt::halo(PP, FF, recv_counts);

    dSync();
    sendF(tc.ranks, tt, recv_counts, FF_pi); /* (sic) */
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
