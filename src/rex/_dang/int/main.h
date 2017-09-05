#define O(p, n) {dSync(); dbg::check_pos_pu(p, n, __FILE__, __LINE__, ""); dSync();}
namespace rex {

enum {OK, FAIL};
static int check_one(Particle p) {
    enum {X, Y, Z};
    float *r;
    r = p.r;
    if (isnan(r[X])) return FAIL;
    return OK;
}
static int check_hst0(Particle *pp, int n) {
    int i;
    for (i = 0; i < n; i++)
        if (check_one(pp[i]) != OK) return FAIL;
    return OK;
}
static int check_hst(Pap26 PP, int counts[26]) {
    int n, i;
    for (i = 0; i < 26; i++) {
        n = counts[i];
        if (check_hst0(PP.d[i], n) != OK) return FAIL;
    }
    return OK;
}
static void report_hst() {
    assert(0);
}

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
    if (check_hst(PP_pi, recv_counts) != OK) report_hst();

    if (!first) s::waitA(); else first = 0;

    dSync();
    for (int i = 0; i < 26; i++) {
        MSG("recv_counts[%d]: %d/%d", i, recv_counts[i], MAX_OBJ_DENSITY*frag_ncell(i));
        O(PP.d[i], recv_counts[i]);
    }
    if (fsiforces)     fsi::halo(PP, FF, recv_counts);
    dSync();
    if (contactforces) cnt::halo(PP, FF, recv_counts);
    dSync();

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
